from django.db import models
import uuid
import secrets
from dashboard.models import CustomUser
from utils.fields import EncryptedCharField


class EdgeDevice(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    DEVICE_TYPE_CHOICES = [
        ('pir_sensor', 'PIR Sensor'),
        ('smart_plug', 'Smart Plug'),
        ('smart_hub', 'Smart Hub'),
    ]
    # Peripherals registered via hub POST /api/hub/sync/ must use one of these (hub itself is not synced here).
    HUB_SYNC_DEVICE_TYPES = frozenset({"pir_sensor", "smart_plug"})

    hub_device = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='devices')
    device_type = models.CharField(max_length=20, choices=DEVICE_TYPE_CHOICES, null=True, blank=True)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='devices', null=True, blank=True)
    battery_level = models.IntegerField(default=100, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    LOCATION_CHOICES = [
        ('living_room', 'Living Room'),
        ('bedroom', 'Bedroom'),
        ('kitchen', 'Kitchen'),
        ('bathroom', 'Bathroom'),
        ('office', 'Office'),
        ('garage', 'Garage'),
    ]
    location = models.CharField(max_length=20, choices=LOCATION_CHOICES, default='living_room', null=True, blank=True)
    SPECIAL_USE_CHOICES = [
        ('medicine_cabinet', 'Medicine Cabinet'),
    ]
    special_use = models.CharField(max_length=20, choices=SPECIAL_USE_CHOICES, null=True, blank=True)

    serial_number = models.CharField(max_length=50, unique=True)
    api_key = EncryptedCharField(max_length=500, default=secrets.token_urlsafe, unique=True, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    is_provisioned = models.BooleanField(default=False) # if True, the device has been provisioned with a user's credentials 

    class Meta:
        db_table = 'api_edge_device'

class EdgeEvent(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    ACTION_CHOICES = [
        ('tv', 'TV'),
        ('medicine', 'Medicine'),
        ('reaching', 'Reaching'),
        ("unknown", "Unknown")
    ]
    action = models.CharField(max_length=20, choices=ACTION_CHOICES, default='unknown', null=True, blank=True)
    device_state = models.JSONField(default=dict)
    keypoints = models.JSONField(default=dict)
    hub_device = models.ForeignKey(EdgeDevice, on_delete=models.CASCADE, related_name='events')
    # Device that triggered the event (e.g. PIR that fired). Same shape as device_state items: device_type, serial_number, ...
    trigger_device = models.JSONField(null=True, blank=True)
    
    # Event Metadata for calculation observation. 
    inference_result = models.JSONField(default=list, blank=True)
    POSE_CLASSIFICATION_CHOICES = [
        ('lying', 'Lying'),
        ('reaching', 'Reaching'),
        ('sitting', 'Sitting'),
        ('standing', 'Standing'),
        ('unknown', 'Unknown')
    ]
    pose_classification = models.CharField(max_length=20, choices=POSE_CLASSIFICATION_CHOICES, default='unknown', null=True, blank=True)
    
    is_keypoints_normalized = models.BooleanField(default=False)
    is_processed = models.BooleanField(default=False) # AI background process to figure out alerts
    is_alert = models.BooleanField(default=False)
    alert_reasoning = models.TextField(null=True, blank=True)
    is_resolved = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'api_edge_event'
        
        
