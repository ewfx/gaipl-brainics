async function handleSend() {
    const input = document.getElementById('user-input');
    const message = input.value;
    displayMessage(message, 'user');
    input.value = '';

    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    });
    const data = await response.json();
    displayMessage(data.chatbot_response, 'bot');
}

function displayMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const msgDiv = document.createElement('div');
    msgDiv.className = sender;
    msgDiv.innerText = message;
    chatBox.appendChild(msgDiv);
}
