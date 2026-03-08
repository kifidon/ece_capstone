import base64

from cryptography.fernet import Fernet
from django.conf import settings
from django.db import models


def _get_fernet():
    key = getattr(settings, "FIELD_ENCRYPTION_KEY", None)
    if not key:
        raise ValueError(
            "settings.FIELD_ENCRYPTION_KEY is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    if isinstance(key, str):
        key = key.encode()
    return Fernet(key)


class EncryptedCharField(models.CharField):
    """
    A CharField that transparently encrypts values before saving to the DB
    and decrypts when reading. Uses Fernet (AES-128-CBC + HMAC-SHA256).

    Values are stored as base64-encoded ciphertext.
    Null/blank values pass through unencrypted.
    """

    def get_prep_value(self, value):
        if value is None or value == "":
            return value
        f = _get_fernet()
        return f.encrypt(value.encode()).decode()

    def from_db_value(self, value, expression, connection):
        if value is None or value == "":
            return value
        try:
            f = _get_fernet()
            return f.decrypt(value.encode()).decode()
        except Exception:
            return value

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        path = "utils.fields.EncryptedCharField"
        return name, path, args, kwargs
