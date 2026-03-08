from functools import wraps

from flask import jsonify, request


def require_api_key(hub_state):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            key = request.headers.get("X-API-Key")
            if not hub_state["api_key"] or key != hub_state["api_key"]:
                return jsonify({"error": "Unauthorized"}), 401
            return f(*args, **kwargs)
        return decorated
    return decorator
