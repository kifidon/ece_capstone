from rest_framework import serializers
from .models import EdgeDevice, EdgeEvent

class EdgeDeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeDevice
        fields = '__all__'

class EdgeEventSerializer(serializers.ModelSerializer):
    devices = EdgeDeviceSerializer(many=True, read_only=True)
    
    class Meta:
        model = EdgeEvent
        fields = '__all__'
        
    def create(): 
        pass        