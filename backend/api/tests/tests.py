import json
import os

from django.test import TestCase
from django.forms.models import model_to_dict
from api.models import EdgeDevice, EdgeEvent
from api.tasks import process_event
from dashboard.models import CustomUser


# Fixture: device_state as sent from hub (same shape as to_registration_payload).
# EdgeEventProcessor rules use d["device_type"] and d.get("is_on"), etc.
DEVICE_STATE_FIXTURE = [
    {
        "device_type": "smart_plug",
        "serial_number": "80068E3A1B2C3D4E",
        "battery_level": None,
        "last_seen": 1710123458.123,
        "alias": "Living Room Lamp",
        "is_on": True,
    },
    {
        "device_type": "pir_sensor",
        "serial_number": "PIR-D4E5F6",
        "battery_level": 88,
        "last_seen": 1710123440.456,
        "alias": None,
        "is_on": None,
    },
    {
        "device_type": "pir_sensor",
        "serial_number": "PIR-789ABC",
        "battery_level": 91,
        "last_seen": 1710123435.789,
        "alias": None,
        "is_on": None,
    },
]

# Keypoints from lying04 clip. Format: {"frame_0": [[y, x, conf], ... (17 keypoints)], ...}
with open(os.path.join(os.path.dirname(__file__), "lying04.json"), "r") as f:
    KEYPOINTS_FIXTURE = json.load(f)


class ProcessEventTestCase(TestCase):
    """Test process_event task: real inference run, pose classified as sitting."""

    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123",
        )
        self.hub = EdgeDevice.objects.create(
            type="smart_hub",
            serial_number="HUB-001",
            user=self.user,
        )
        self.event = EdgeEvent.objects.create(
            hub_device=self.hub,
            device_state=DEVICE_STATE_FIXTURE,
            keypoints=KEYPOINTS_FIXTURE,
        )

    def test_process_event_runs_inference_and_classifies(self):
        process_event(str(self.event.id))

        self.event.refresh_from_db()
        output = model_to_dict(self.event)
        
        print(output["inference_result"])
        print(output["pose_classification"])
        print(output["is_keypoints_normalized"])
        print(output["action"])


        self.assertTrue(self.event.is_keypoints_normalized)
        self.assertIsNotNone(self.event.inference_result)
        self.assertIsNotNone(self.event.action)
