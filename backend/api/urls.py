from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import EdgeEventView, EdgeDeviceView, device_register, device_claim, register_discovered_devices, hub_config

router = DefaultRouter()
router.register(r'events', EdgeEventView, basename='edge-event')
router.register(r'devices', EdgeDeviceView, basename='edge-device')

urlpatterns = [
    path('', include(router.urls)),
    path('devices/register/', device_register, name='device-register'),
    path('devices/claim/', device_claim, name='device-claim'),
    path('devices/sync/', register_discovered_devices, name='device-sync'),
    path('devices/<str:serial_number>/config/', hub_config, name='hub-config'),
]
