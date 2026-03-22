from rest_framework import serializers
from .models import EdgeDevice, EdgeEvent


class EdgeDeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeDevice
        fields = '__all__'


class EdgeEventSerializer(serializers.ModelSerializer):
    # Hub sends "devices" (list); we map to device_state when creating
    devices = serializers.ListField(child=serializers.DictField(), write_only=True, required=False)

    class Meta:
        model = EdgeEvent
        fields = [
            'id', 'timestamp', 'action', 'device_state', 'devices', 'keypoints',
            'hub_device', 'trigger_device', 'pose_classification', 'inference_result',
            'is_processed', 'is_alert', 'is_resolved', 'is_deleted', 'is_keypoints_normalized',
        ]
        read_only_fields = [
            'id', 'timestamp', 'action', 'pose_classification',
            'inference_result', 'is_processed', 'is_alert', 'is_keypoints_normalized',
            'device_state',
        ]
        extra_kwargs = {
            'hub_device': {'required': True},
            'device_state': {'required': False},
            'keypoints': {'required': True},
            'trigger_device': {'required': False},
        }

    def validate_hub_device(self, value):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            if value.user_id != request.user.id:
                raise serializers.ValidationError("Hub does not belong to you.")
        return value

    def create(self, validated_data):
        devices = validated_data.pop('devices', None)
        if devices is not None:
            validated_data['device_state'] = devices
        if validated_data.get('device_state') is None:
            validated_data['device_state'] = []
        return super().create(validated_data)