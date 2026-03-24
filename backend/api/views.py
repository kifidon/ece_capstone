import logging

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from .models import EdgeEvent, EdgeDevice
from .serializers import EdgeEventSerializer, EdgeDeviceSerializer
from .tasks import process_event, push_config_to_hub, run_post_process_events
from .decorators import require_hub_api_key

logger = logging.getLogger(__name__)


class EdgeEventView(ModelViewSet):
    serializer_class = EdgeEventSerializer

    def get_permissions(self):
        # Hub POSTs events without JWT; browser app uses JWT for list/detail/update/delete
        if self.action == 'create':
            return [AllowAny()]
        return [IsAuthenticated()]

    def get_queryset(self):
        qs = EdgeEvent.objects.filter(is_deleted=False).order_by('-timestamp')
        if self.request.user.is_authenticated:
            # Events may reference the smart_hub or a peripheral. Peripherals sometimes have
            # user=NULL while the parent hub is claimed; include any device we own or that
            # hangs off our hub so those events still list correctly.
            user = self.request.user
            device_ids = EdgeDevice.objects.filter(
                Q(user=user) | Q(hub_device__user=user)
            ).values_list('id', flat=True)
            qs = qs.filter(hub_device_id__in=device_ids)
        return qs

    def perform_create(self, serializer):
        event = serializer.save()
        process_event.delay(str(event.id))

    def destroy(self, request, *args, **kwargs):
        event = self.get_object()
        event.is_deleted = True
        event.save(update_fields=['is_deleted'])
        return Response(status=status.HTTP_204_NO_CONTENT)

class EdgeDeviceView(ModelViewSet):
    serializer_class = EdgeDeviceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return EdgeDevice.objects.filter(user=self.request.user)


@api_view(["POST"])
@permission_classes([AllowAny])
def device_register(request):
    """
    Hub calls this on boot (after WiFi setup) with its serial number.
    Updates the hub's IP address. If already claimed by a user, pushes config immediately.

    POST /api/devices/register/
    Body: { "serial_number": "..." }
    """
    serial_number = request.data.get("serial_number")
    if not serial_number:
        return JsonResponse({"error": "serial_number is required"}, status=400)

    device = get_object_or_404(EdgeDevice, serial_number=serial_number)

    client_ip = request.META.get("HTTP_X_FORWARDED_FOR", request.META.get("REMOTE_ADDR"))
    if client_ip and "," in client_ip: # if the request is coming from a proxy, get the first IP address
        client_ip = client_ip.split(",")[0].strip()

    device.ip_address = client_ip
    device.is_active = True
    device.save(update_fields=["ip_address", "is_active"])

    if device.user:
        push_config_to_hub.delay(str(device.id))
        return JsonResponse({
            "status": "claimed",
            "hub_device_id": str(device.id),
            "message": "Config push scheduled.",
        })

    return JsonResponse({
        "status": "waiting_for_claim",
        "hub_device_id": str(device.id),
        "message": "Device registered. Waiting for user to claim via website.",
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def device_claim(request):
    """
    Frontend calls this when a user enters a device serial number.
    Links the device (and child devices if it's a hub) to the user.
    If it's a hub, pushes config to it.

    POST /api/devices/claim/
    Body: { "serial_number": "..." }
    """
    serial_number = request.data.get("serial_number")
    if not serial_number:
        return JsonResponse({"error": "serial_number is required"}, status=400)

    device = get_object_or_404(EdgeDevice, serial_number=serial_number)

    if device.user and device.user != request.user:
        return JsonResponse({"error": "Device is already claimed by another user."}, status=409)

    device.user = request.user
    device.save(update_fields=["user"])

    if device.device_type == "smart_hub":
        device.devices.update(user=request.user)

        push_config_to_hub.delay(str(device.id))
        return JsonResponse({
            "status": "claimed",
            "config_push": "scheduled",
            "device": EdgeDeviceSerializer(device).data,
        })

    return JsonResponse({
        "status": "claimed",
        "device": EdgeDeviceSerializer(device).data,
    })


@api_view(["POST"])
@permission_classes([AllowAny])
@require_hub_api_key(serial_field="hub_serial")
def register_discovered_devices(request, hub):
    """
    Hub sends its discovered devices (PIR sensors, Kasa plugs) to be registered.
    Creates new EdgeDevice records linked to the hub, or updates existing ones.

    POST /api/devices/sync/
    Headers: X-API-Key: <hub's api key>
    Body: {
        "hub_serial": "...",
        "devices": [
            {"serial_number": "...", "device_type": "pir_sensor", "battery_level": 95},
            {"serial_number": "...", "device_type": "smart_plug"},
        ]
    }
    Each device must include serial_number and device_type (pir_sensor or smart_plug). Missing or invalid values return 400.
    """
    devices = request.data.get("devices", [])
    if not isinstance(devices, list):
        return JsonResponse({"error": "devices must be a JSON array"}, status=400)

    allowed = EdgeDevice.HUB_SYNC_DEVICE_TYPES
    device_errors = []
    for i, dev_data in enumerate(devices):
        if not isinstance(dev_data, dict):
            device_errors.append({"index": i, "detail": "each entry must be an object"})
            continue
        serial = dev_data.get("serial_number")
        dtype = dev_data.get("device_type")
        row_msgs = []
        if not serial or not str(serial).strip():
            row_msgs.append("serial_number is required")
        if dtype is None or (isinstance(dtype, str) and not dtype.strip()):
            row_msgs.append("device_type is required")
        elif dtype not in allowed:
            row_msgs.append(
                f"device_type must be one of: {sorted(allowed)} (got {dtype!r})"
            )
        if row_msgs:
            device_errors.append({"index": i, "detail": "; ".join(row_msgs)})

    if device_errors:
        return JsonResponse(
            {
                "error": "invalid devices payload",
                "device_errors": device_errors,
            },
            status=400,
        )

    created = updated = 0
    for dev_data in devices:
        serial = str(dev_data["serial_number"]).strip()
        dtype = dev_data["device_type"]
        if isinstance(dtype, str):
            dtype = dtype.strip()

        device, was_created = EdgeDevice.objects.update_or_create(
            serial_number=serial,
            defaults={
                "device_type": dtype,
                "hub_device": hub,
                "user": hub.user,
                "battery_level": dev_data.get("battery_level"),
                "is_active": True,
            },
        )

        if was_created:
            created += 1
        else:
            updated += 1

    logger.info(f"Device sync from hub {hub.serial_number}: {created} created, {updated} updated")
    return JsonResponse({"created": created, "updated": updated})


@api_view(["GET"])
@permission_classes([AllowAny])
def hub_config(request, serial_number):
    """
    Fallback endpoint for hub to fetch its config.
    Primary flow is push-based via push_config_to_hub.
    """
    hub = get_object_or_404(EdgeDevice, serial_number=serial_number, device_type="smart_hub")

    if not hub.user:
        return JsonResponse({"error": "Hub not claimed by any user"}, status=404)

    return JsonResponse({
        "api_key": hub.api_key,
        "hub_device_id": str(hub.id),
        "kasa_username": hub.user.kasa_username,
        "kasa_password": hub.user.kasa_password,
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def trigger_post_process(request):
    """
    TEMP: public test hook — enqueues ``run_post_process_events``. Remove before production.
    """
    run_post_process_events.delay()
    logger.info("run_post_process_events queued via trigger_post_process (public test endpoint)")
    return Response(
        {"status": "queued", "task": "api.tasks.run_post_process_events"},
        status=status.HTTP_202_ACCEPTED,
    )
