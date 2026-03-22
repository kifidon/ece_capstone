"""
Celery configuration for the config project.
"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Required so autodiscover sees INSTALLED_APPS and registers tasks in api.tasks, etc.
import django

django.setup()

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
