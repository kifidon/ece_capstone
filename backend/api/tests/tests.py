import json
import os

from django.test import TestCase
from django.forms.models import model_to_dict
from django.utils import timezone
from datetime import datetime, timedelta, time
import unittest

from api.models import EdgeDevice, EdgeEvent
from api.tasks import process_event, EdgeEventProcessor
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


class PostProcessAnomalyTestCase(TestCase):
    """
    Baseline: person watches TV once a day between 2-4pm.
    Test candidates:
      - 3:30pm on the same day => not anomalous
      - 5:00pm on the same day => anomalous (outside 2-4pm window)
      - 3:00am on the next day => anomalous (far outside daily pattern)
    """

    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username="anomalyuser",
            email="anomaly@example.com",
            password="testpass123",
        )
        self.hub = EdgeDevice.objects.create(
            type="smart_hub",
            serial_number="HUB-ANOM-001",
            user=self.user,
        )

        local_now = timezone.localtime(timezone.now())
        tz = timezone.get_current_timezone()
        today = local_now.date()
        tomorrow = (local_now + timedelta(days=1)).date()

        def make_dt(d, t: time) -> datetime:
            naive = datetime.combine(d, t)
            return timezone.make_aware(naive, tz)

        # Baseline: 5 events in last 7 days spread across 3pm-5pm (exclusive),
        # so 3:30pm is clearly part of "normal" timing while 5:00pm stays outside.
        baseline_times = [
            (15, 0),   # 3:00pm
            (15, 30),  # 3:30pm
            (16, 0),   # 4:00pm
            (16, 30),  # 4:30pm
            (16, 45),  # 4:45pm
        ]
        for days_ago, (hour, minute) in zip(range(1, 6), baseline_times):
            dt = local_now - timedelta(days=days_ago)
            dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            ev = EdgeEvent.objects.create(
                hub_device=self.hub,
                action="tv",
                pose_classification="sitting",
                is_processed=True,
                is_alert=False,
                alert_reasoning=None,
            )
            EdgeEvent.objects.filter(id=ev.id).update(timestamp=dt)

        # Candidates: 3:30pm, 5:00pm, 3:00am
        self.event_ok = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="tv",
            pose_classification="sitting",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_ok.id).update(timestamp=make_dt(today, time(15, 30)))

        self.event_anom_1 = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="tv",
            pose_classification="sitting",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_anom_1.id).update(timestamp=make_dt(today, time(17, 0)))

        self.event_anom_2 = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="tv",
            pose_classification="sitting",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_anom_2.id).update(timestamp=make_dt(tomorrow, time(3, 0)))

    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not set")
    def test_post_process_event_marks_anomalies(self):
        processor = EdgeEventProcessor()
        processor.post_process_event()

        self.event_ok.refresh_from_db()
        self.event_anom_1.refresh_from_db()
        self.event_anom_2.refresh_from_db()

        # All candidates should be marked processed.
        self.assertTrue(self.event_ok.is_processed)
        self.assertTrue(self.event_anom_1.is_processed)
        self.assertTrue(self.event_anom_2.is_processed)

        # Only anomalies should be flagged.
        self.assertFalse(self.event_ok.is_alert)
        self.assertTrue(self.event_anom_1.is_alert)
        self.assertTrue(self.event_anom_2.is_alert)
        self.assertIsNotNone(self.event_anom_1.alert_reasoning)
        self.assertIsNotNone(self.event_anom_2.alert_reasoning)


class PostProcessMedicineAnomalyTestCase(TestCase):
    """
    Baseline: medicine grab once per day for a week between 8am-12pm.
    Test candidates:
      - one medicine grab (normal)
      - one day with 3 medicine grabs (expected anomalous)
    """

    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username="medicineuser",
            email="medicine@example.com",
            password="testpass123",
        )
        self.hub = EdgeDevice.objects.create(
            type="smart_hub",
            serial_number="HUB-MED-001",
            user=self.user,
        )

        local_now = timezone.localtime(timezone.now())
        tz = timezone.get_current_timezone()
        today = local_now.date()

        def make_dt(d, t: time) -> datetime:
            naive = datetime.combine(d, t)
            return timezone.make_aware(naive, tz)

        # Baseline: 7 events within the last 7 days, each day exactly once between 8am-12pm.
        baseline_times = [
            (8, 15),
            (9, 30),
            (10, 0),
            (10, 45),
            (11, 0),
            (11, 30),
            (12, 0),
        ]
        for days_ago, (hour, minute) in zip(range(1, 8), baseline_times):
            dt = local_now - timedelta(days=days_ago)
            dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            ev = EdgeEvent.objects.create(
                hub_device=self.hub,
                action="medicine",
                pose_classification="standing",
                is_processed=True,
                is_alert=False,
                alert_reasoning=None,
            )
            EdgeEvent.objects.filter(id=ev.id).update(timestamp=dt)


        self.event_ok = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="medicine",
            pose_classification="standing",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_ok.id).update(
            timestamp=make_dt(today, time(10, 15))
        )

        self.event_med_1 = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="medicine",
            pose_classification="standing",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_med_1.id).update(
            timestamp=make_dt(today, time(9, 0))
        )

        self.event_med_2 = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="medicine",
            pose_classification="standing",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_med_2.id).update(
            timestamp=make_dt(today, time(10, 45))
        )

        self.event_med_3 = EdgeEvent.objects.create(
            hub_device=self.hub,
            action="medicine",
            pose_classification="standing",
            is_processed=False,
            is_alert=False,
            alert_reasoning=None,
        )
        EdgeEvent.objects.filter(id=self.event_med_3.id).update(
            timestamp=make_dt(today, time(11, 30))
        )

    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not set")
    def test_post_process_medicine_frequency_anomalies(self):
        processor = EdgeEventProcessor()
        processor.post_process_event()

        self.event_ok.refresh_from_db()
        self.event_med_1.refresh_from_db()
        self.event_med_2.refresh_from_db()
        self.event_med_3.refresh_from_db()

        # All candidates should be marked processed.
        self.assertTrue(self.event_ok.is_processed)
        self.assertTrue(self.event_med_1.is_processed)
        self.assertTrue(self.event_med_2.is_processed)
        self.assertTrue(self.event_med_3.is_processed)

        # All 4 events happen on the same day where medicine is grabbed multiple times,
        # so the model should flag all of them as anomalies.
        self.assertTrue(self.event_ok.is_alert)
        self.assertTrue(self.event_med_1.is_alert)
        self.assertTrue(self.event_med_2.is_alert)
        self.assertTrue(self.event_med_3.is_alert)

        self.assertIsNotNone(self.event_ok.alert_reasoning)
        self.assertIsNotNone(self.event_med_1.alert_reasoning)
        self.assertIsNotNone(self.event_med_2.alert_reasoning)
        self.assertIsNotNone(self.event_med_3.alert_reasoning)
