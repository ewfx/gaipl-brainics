from flask import Blueprint, request, jsonify
from .chatbot import get_chatbot_response

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    response = get_chatbot_response(message)
    return jsonify({"response": response})
