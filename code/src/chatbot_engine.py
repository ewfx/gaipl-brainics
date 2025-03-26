# chatbot_engine.py
from transformers import pipeline

class ChatbotEngine:
    def __init__(self):
        self.chat = pipeline("text-generation", model="gpt2")

    def chat_response(self, context):
        response = self.chat(context, max_length=100, do_sample=True, temperature=0.7)
        return response[0]['generated_text']
