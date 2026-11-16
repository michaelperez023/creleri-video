function showVideoModal() {
    const loader = document.getElementById('videoLoader');
    loader.style.display = 'flex';
    loader.style.flexDirection = 'row';
}

function hideVideoModal() {
    document.getElementById('videoLoader').style.display = 'none';
}

// The paperclip label is the visible control, but the visually hidden input behind it is
// still keyboard-focusable, so it is disabled as well. The segmentation mode is fixed per batch.
function setUploadEnabled(enabled) {
    const label = document.getElementById('file-label');
    label.classList.toggle('disabled', !enabled);
    label.style.opacity = enabled ? '1' : '0.5';
    document.getElementById('file-upload').disabled = !enabled;
    document.getElementById('segmentation-type-select').disabled = !enabled;
}

// Toggles the concept visualization box, or forces it open or closed with 'Expand' / 'Collapse'.
function conceptVisualizationCollapsable(action) {
    const box = document.getElementById('concept-visualization-box');
    const button = document.getElementById('concept-visualization-collapsible-btn');
    const expand = action ? action === 'Expand' : box.style.display !== '';

    box.style.display = expand ? '' : 'none';
    box.classList.toggle('expanded', expand);
    button.setAttribute('aria-expanded', String(expand));
    button.querySelector('.material-symbols-outlined').textContent =
        expand ? 'expand_circle_up' : 'expand_circle_down';
}

function displayMessage(message, className) {
    const conversationArea = document.getElementById('conversation-area');
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.classList.add('chat-message', className);
    conversationArea.appendChild(messageDiv);
    conversationArea.scrollTop = conversationArea.scrollHeight;
}

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('file-upload').addEventListener('change', handleFileUpload);

    // Grow the message box with its content, up to four rows.
    document.getElementById('user-input').addEventListener('keyup', function () {
        this.rows = 1;
        this.rows = Math.min(4, this.scrollHeight / 32);
    });
});
