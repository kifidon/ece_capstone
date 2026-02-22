from rest_framework import serializers
from .models import Device, Event

class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = '__all__'

class EventSerializer(serializers.ModelSerializer):
    devices = DeviceSerializer(many=True, read_only=True)
    
    class Meta:
        model = Event
        fields = '__all__'