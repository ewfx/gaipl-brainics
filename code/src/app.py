from flask import Flask, render_template, request, jsonify
from chatbot import generate_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Flask will look inside /templates/

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    bot_response = generate_response(user_input)
    return jsonify({'chatbot_response': bot_response})
print(app.url_map)
if __name__ == '__main__':
    app.run(debug=True, port=5050)
