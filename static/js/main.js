/* ══════════════════════════════════════
   PawScan AI — Main JavaScript
   ══════════════════════════════════════ */

// ──── Image Upload with Drag & Drop ────
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadIcon = document.getElementById('uploadIcon');
    const uploadTitle = document.getElementById('uploadTitle');
    const uploadHint = document.getElementById('uploadHint');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (dropZone && imageInput) {
        // Click to browse
        dropZone.addEventListener('click', () => imageInput.click());

        // Drag events
        ['dragenter', 'dragover'].forEach(event => {
            dropZone.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            });
        });

        // Handle drop
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                imageInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        // Handle file selection
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadIcon.style.display = 'none';
                uploadTitle.textContent = file.name;
                uploadHint.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB • Click to change`;
                dropZone.classList.add('has-image');
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }

    // ──── Animate elements on scroll ────
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.6s ease-out forwards';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.step-card, .stat-card, .info-card, .disease-item').forEach(el => {
        el.style.opacity = '0';
        observer.observe(el);
    });
});


// ──── Chat Widget ────
let chatOpen = false;

function toggleChat() {
    const chatWindow = document.getElementById('chatWindow');
    const chatToggleIcon = document.getElementById('chatToggleIcon');

    if (!chatWindow) return;

    chatOpen = !chatOpen;

    if (chatOpen) {
        chatWindow.classList.add('open');
        chatToggleIcon.textContent = '✕';
        document.getElementById('chatInput').focus();
    } else {
        chatWindow.classList.remove('open');
        chatToggleIcon.textContent = '🤖';
    }
}

function handleChatKeypress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function sendQuickMessage(message) {
    const chatInput = document.getElementById('chatInput');
    chatInput.value = message;
    sendMessage();
}

async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const message = chatInput.value.trim();

    if (!message) return;

    // Add user message
    const userBubble = document.createElement('div');
    userBubble.className = 'chat-message user-message';
    userBubble.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;
    chatMessages.appendChild(userBubble);

    // Clear input
    chatInput.value = '';

    // Remove quick actions after first message
    const quickActions = chatMessages.querySelector('.chat-quick-actions');
    if (quickActions) quickActions.remove();

    // Add typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bot-message';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🐾</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        // Send to chatbot API
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // Remove typing indicator
        const typing = document.getElementById('typingIndicator');
        if (typing) typing.remove();

        // Add bot response
        const botBubble = document.createElement('div');
        botBubble.className = 'chat-message bot-message';
        botBubble.innerHTML = `
            <div class="message-avatar">🐾</div>
            <div class="message-content">
                <p>${formatBotResponse(data.response)}</p>
            </div>
        `;
        chatMessages.appendChild(botBubble);

    } catch (error) {
        // Remove typing indicator
        const typing = document.getElementById('typingIndicator');
        if (typing) typing.remove();

        // Add error message
        const errorBubble = document.createElement('div');
        errorBubble.className = 'chat-message bot-message';
        errorBubble.innerHTML = `
            <div class="message-avatar">🐾</div>
            <div class="message-content">
                <p>Sorry, I'm having trouble connecting. Please try again! 🙏</p>
            </div>
        `;
        chatMessages.appendChild(errorBubble);
    }

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBotResponse(text) {
    // Convert markdown-like formatting
    let formatted = escapeHtml(text);

    // Bold text: *text*
    formatted = formatted.replace(/\*([^*]+)\*/g, '<strong>$1</strong>');

    // Line breaks
    formatted = formatted.replace(/\n\n/g, '</p><p>');
    formatted = formatted.replace(/\n/g, '<br>');

    // Bullet points
    formatted = formatted.replace(/• /g, '• ');

    return formatted;
}
