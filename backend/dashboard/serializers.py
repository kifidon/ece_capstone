from rest_framework import serializers
from .models import CustomUser
from api.serializers import DeviceSerializer

class CustomUserSerializer(serializers.ModelSerializer):
    devices = DeviceSerializer(many=True, read_only=True)
    
    class Meta:
        model = CustomUser
        fields = '__all__'