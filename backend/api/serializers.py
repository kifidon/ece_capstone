from rest_framework import serializers
from .models import EdgeDevice, EdgeEvent


class EdgeDeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeDevice
        fields = '__all__'


class EdgeEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeEvent
        fields = [
            'id', 'timestamp', 'action', 'device_state', 'keypoints',
            'hub_device', 'pose_classification', 'inference_result',
            'is_processed', 'is_alert', 'is_resolved', 'is_deleted', 'is_keypoints_normalized',
        ]
        read_only_fields = [
            'id', 'timestamp', 'action', 'pose_classification',
            'inference_result', 'is_processed', 'is_alert', 'is_keypoints_normalized', "device_state", "keypoints", "hub_device",
        ]
        extra_kwargs = {
            'hub_device': {'required': True},
            'device_state': {'required': True},
            'keypoints': {'required': True},
        }