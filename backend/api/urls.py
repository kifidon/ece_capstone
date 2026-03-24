from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    EdgeEventView,
    EdgeDeviceView,
    device_register,
    device_claim,
    register_discovered_devices,
    hub_get_config,
    trigger_post_process,
)

router = DefaultRouter()
router.register(r'events', EdgeEventView, basename='edge-event')
router.register(r'devices', EdgeDeviceView, basename='edge-device')

# Hub / provisioning lives under /api/hub/* so it never collides with
# /api/devices/<pk>/ from the router.
urlpatterns = [
    path('hub/register/', device_register, name='device-register'),
    path('hub/claim/', device_claim, name='device-claim'),
    path('hub/config/', hub_get_config, name='hub-get-config'),
    path('hub/sync/', register_discovered_devices, name='device-sync'),
    path('tasks/run-post-process/', trigger_post_process, name='trigger-post-process'),
    path('', include(router.urls)),
]
