import logging
from flask import Blueprint, request, redirect, render_template

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

@api_bp.route("/motion-detected", methods=["POST"])
def motion_detected():
    data = request.get_json()
    