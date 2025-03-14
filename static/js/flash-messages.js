// Function to handle flash messages
function setupFlashMessages() {
    // Get all flash messages
    const flashMessages = document.querySelectorAll('.flash-message');
    
    // Add show class to make them visible with animation
    flashMessages.forEach((message, index) => {
        // Stagger the animations slightly for multiple messages
        setTimeout(() => {
            message.classList.add('show');
        }, index * 150);
        
        // Add close button functionality
        const closeBtn = message.querySelector('.close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                message.classList.remove('show');
                setTimeout(() => {
                    message.remove();
                }, 300); // Match this with CSS transition time
            });
        }
        
        // Set timeout to automatically remove after 5 seconds
        setTimeout(() => {
            message.classList.remove('show');
            setTimeout(() => {
                message.remove();
            }, 300); // Match this with CSS transition time
        }, 5000); // Display for 5 seconds
    });
    // Create a special case for AI status messages that shouldn't auto-hide
    const regularMessages = document.querySelectorAll('.flash-message:not(.ai-status-message)');
    regularMessages.forEach((message, index) => {
        // Your existing animation code...
    
        // Auto-hide after 5 seconds - keep this logic only for regular messages
        setTimeout(() => {
            message.classList.remove('show');
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });
}

// Run when DOM is loaded
document.addEventListener('DOMContentLoaded', setupFlashMessages);
// Create a function to show model status as a temporary flash message
window.showModelStatus = function(status, isReady) {
    const container = document.querySelector('.flash-messages-container');
    if (!container) return;
    
    const messageClass = isReady ? 'success' : 'info';
    const icon = isReady ? 'check-circle' : 'info-circle';
    
    const messageHTML = `
        <div class="flash-message ${messageClass}">
            <div class="icon">
                <i class="fas fa-${icon}"></i>
            </div>
            <div class="message-content">${status}</div>
            <button class="close-btn" aria-label="Close">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', messageHTML);
    setupFlashMessages();
};