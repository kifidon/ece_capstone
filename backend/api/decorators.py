from functools import wraps

from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from .models import EdgeDevice


def require_hub_api_key(serial_field="hub_serial"):
    """
    Verify the X-API-Key header matches the hub's stored API key.
    Looks up the hub using the serial number from request.data[serial_field].
    Passes the hub as the second argument to the wrapped view.
    """
    def decorator(f):
        @wraps(f)
        def wrapped(request, *args, **kwargs):
            hub_serial = request.data.get(serial_field)
            if not hub_serial:
                return JsonResponse({"error": f"{serial_field} is required"}, status=400)

            hub = get_object_or_404(EdgeDevice, serial_number=hub_serial, device_type="smart_hub")

            api_key = request.headers.get("X-API-Key")
            if not api_key or api_key != hub.api_key:
                return JsonResponse({"error": "Unauthorized"}, status=401)

            return f(request, hub, *args, **kwargs)
        return wrapped
    return decorator
