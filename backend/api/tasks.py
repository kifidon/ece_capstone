"""
Example Celery tasks for the api app.
Import and call .delay() from views or elsewhere to run async.
"""
from celery import shared_task


@shared_task
def example_task(message: str) -> str:
    """Example task: returns the message after a delay."""
    return f"Processed: {message}"
