// ========== LÓGICA DEL CHAT (GEMINI) ==========
const chatBox = document.getElementById('chatBox');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');

async function sendChatMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    // Mostrar mensaje del usuario
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-msg user';
    userDiv.textContent = '👤 ' + msg;
    chatBox.appendChild(userDiv);
    chatInput.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await response.json();
        const botDiv = document.createElement('div');
        botDiv.className = 'chat-msg bot';
        if (data.error) {
            botDiv.textContent = '⚠️ ' + data.error;
        } else {
            botDiv.textContent = '🤖 ' + data.response;
        }
        chatBox.appendChild(botDiv);
    } catch (error) {
        const botDiv = document.createElement('div');
        botDiv.className = 'chat-msg bot';
        botDiv.textContent = '⚠️ Error conectando con el servidor.';
        chatBox.appendChild(botDiv);
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}

chatSendBtn.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});