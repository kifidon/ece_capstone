from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    EdgeEventView,
    EdgeDeviceView,
    device_register,
    device_claim,
    register_discovered_devices,
    hub_config,
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
    path('hub/sync/', register_discovered_devices, name='device-sync'),
    path('hub/<str:serial_number>/config/', hub_config, name='hub-config'),
    path('tasks/run-post-process/', trigger_post_process, name='trigger-post-process'),
    path('', include(router.urls)),
]
