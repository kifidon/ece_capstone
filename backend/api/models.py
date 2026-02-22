from django.db import models
import uuid
from dashboard.models import CustomUser
# Create your models here.

class Device(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    TYPE_CHOICES = [
        ('pir_sensor', 'PIR Sensor'),
        ('smart_plug', 'Smart Plug'),
        ('smart_hub', 'Smart Hub'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='pir_sensor')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    battery_level = models.IntegerField(default=100, null=True, blank=True, min_value=0, max_value=100)
    is_active = models.BooleanField(default=True)
    LOCATION_CHOICES = [
        ('living_room', 'Living Room'),
        ('bedroom', 'Bedroom'),
        ('kitchen', 'Kitchen'),
        ('bathroom', 'Bathroom'),
        ('office', 'Office'),
        ('garage', 'Garage'),
    ]
    location = models.CharField(max_length=20, choices=LOCATION_CHOICES, default='living_room')

    class Meta:
        db_table = 'api_device'

class Event(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField(auto_now_add=True)
    ACTION_CHOICES = [
        ('standing', 'Standing'),
        ('sitting', 'Sitting'),
        ('laying_down', 'Laying Down'),
        ('reaching', 'Reaching'),
    ]
    action = models.CharField(max_length=20, choices=ACTION_CHOICES, default='standing')
    is_processed = models.BooleanField(default=False)
    is_alert = models.BooleanField(default=False)
    device_state = models.JSONField(default=dict)
    hub_device = models.ForeignKey(Device, on_delete=models.CASCADE)
    # inference_result storing a list
    inference_result = models.JSONField(default=list, blank=True)
    
    class Meta:
        db_table = 'api_event'
        
