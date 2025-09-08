// CS336 Transformer Chat Interface JavaScript

class ChatInterface {
    constructor() {
        this.conversationId = null;
        this.isGenerating = false;
        this.currentConfig = {};
        
        this.initializeElements();
        this.setupEventListeners();
        this.startNewConversation();
        this.loadConfig();
    }
    
    initializeElements() {
        // Main elements
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.chatMessages = document.getElementById('chatMessages');
        this.statusText = document.getElementById('statusText');
        this.modelInfo = document.getElementById('modelInfo');
        this.generationTime = document.getElementById('generationTime');
        
        // Header buttons
        this.configBtn = document.getElementById('configBtn');
        this.statsBtn = document.getElementById('statsBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.clearBtn = document.getElementById('clearBtn');
        
        // Configuration elements
        this.maxTokensInput = document.getElementById('maxTokens');
        this.temperatureInput = document.getElementById('temperature');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.topPInput = document.getElementById('topP');
        this.topPValue = document.getElementById('topPValue');
        this.topKInput = document.getElementById('topK');
        this.samplingStrategySelect = document.getElementById('samplingStrategy');
    }
    
    setupEventListeners() {
        // Message input and sending
        this.messageInput.addEventListener('input', () => this.handleInputChange());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Header buttons
        this.configBtn.addEventListener('click', () => this.showConfigModal());
        this.statsBtn.addEventListener('click', () => this.showStatsModal());
        this.exportBtn.addEventListener('click', () => this.exportConversation());
        this.clearBtn.addEventListener('click', () => this.clearConversation());
        
        // Configuration inputs
        this.temperatureInput.addEventListener('input', (e) => {
            this.temperatureValue.textContent = e.target.value;
        });
        
        this.topPInput.addEventListener('input', (e) => {
            this.topPValue.textContent = e.target.value;
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
    }
    
    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    handleInputChange() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText || this.isGenerating;
        
        // Update send button appearance
        if (hasText && !this.isGenerating) {
            this.sendBtn.style.opacity = '1';
        } else {
            this.sendBtn.style.opacity = '0.6';
        }
    }
    
    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendBtn.disabled) {
                this.sendMessage();
            }
        }
    }
    
    async startNewConversation() {
        try {
            const response = await fetch('/api/new_conversation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                this.conversationId = data.conversation_id;
                this.updateStatus('Ready to chat');
            } else {
                this.showError('Failed to start conversation');
            }
        } catch (error) {
            this.showError('Connection error: ' + error.message);
        }
    }
    
    async sendMessage() {
        if (this.isGenerating || !this.messageInput.value.trim()) {
            return;
        }
        
        const message = this.messageInput.value.trim();
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.handleInputChange();
        
        // Add user message to UI
        this.addMessage('user', message);
        
        // Show generating state
        this.setGenerating(true);
        const typingIndicator = this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/send_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            typingIndicator.remove();
            
            if (response.ok && data.status === 'success') {
                this.addMessage('assistant', data.response, data.generation_time);
                this.updateGenerationTime(data.generation_time);
            } else {
                this.showError(data.error || 'Failed to generate response');
            }
        } catch (error) {
            typingIndicator.remove();
            this.showError('Network error: ' + error.message);
        } finally {
            this.setGenerating(false);
        }
    }
    
    addMessage(role, content, generationTime = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        if (generationTime) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-timestamp';
            timeDiv.innerHTML = `
                <span class="generation-time">Generated in ${(generationTime * 1000).toFixed(0)}ms</span>
            `;
            contentDiv.appendChild(timeDiv);
        }
        
        messageDiv.appendChild(contentDiv);
        
        // Insert before the welcome message or at the end
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage && this.chatMessages.children.length === 1) {
            this.chatMessages.insertBefore(messageDiv, welcomeMessage);
        } else {
            this.chatMessages.appendChild(messageDiv);
        }
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.innerHTML = `
            <div class="message-content typing-indicator">
                <span>Assistant is typing</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        
        return typingDiv;
    }
    
    setGenerating(generating) {
        this.isGenerating = generating;
        this.sendBtn.disabled = generating || !this.messageInput.value.trim();
        this.messageInput.disabled = generating;
        
        if (generating) {
            this.updateStatus('Generating...');
            this.sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            this.updateStatus('Ready');
            this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
        
        this.handleInputChange();
    }
    
    updateStatus(status) {
        this.statusText.textContent = status;
    }
    
    updateGenerationTime(time) {
        this.generationTime.textContent = `Last: ${(time * 1000).toFixed(0)}ms`;
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showError(message) {
        console.error('Chat error:', message);
        this.addMessage('assistant', `Error: ${message}`);
        this.updateStatus('Error occurred');
    }
    
    // Configuration Management
    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            this.currentConfig = config;
            this.updateConfigUI(config);
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }
    
    updateConfigUI(config) {
        if (this.maxTokensInput) this.maxTokensInput.value = config.max_new_tokens || 100;
        if (this.temperatureInput) {
            this.temperatureInput.value = config.temperature || 0.8;
            this.temperatureValue.textContent = config.temperature || 0.8;
        }
        if (this.topPInput) {
            this.topPInput.value = config.top_p || 0.9;
            this.topPValue.textContent = config.top_p || 0.9;
        }
        if (this.topKInput) {
            this.topKInput.value = config.top_k || '';
        }
        if (this.samplingStrategySelect) {
            this.samplingStrategySelect.value = config.sampling_strategy || 'top_p';
        }
    }
    
    async saveConfig() {
        const config = {
            max_new_tokens: parseInt(this.maxTokensInput.value) || 100,
            temperature: parseFloat(this.temperatureInput.value) || 0.8,
            top_p: parseFloat(this.topPInput.value) || 0.9,
            top_k: this.topKInput.value ? parseInt(this.topKInput.value) : null,
            sampling_strategy: this.samplingStrategySelect.value || 'top_p',
            do_sample: true
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                this.currentConfig = data.config;
                this.showSuccess('Configuration saved!');
                closeModal('configModal');
            } else {
                this.showError('Failed to save configuration');
            }
        } catch (error) {
            this.showError('Failed to save configuration: ' + error.message);
        }
    }
    
    showConfigModal() {
        this.loadConfig(); // Refresh config
        document.getElementById('configModal').style.display = 'block';
    }
    
    async showStatsModal() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            const statsContent = document.getElementById('statsContent');
            statsContent.innerHTML = '';
            
            const statItems = [
                { label: 'Active Conversations', value: stats.active_conversations },
                { label: 'Total Messages', value: stats.total_messages },
                { label: 'Total Tokens Generated', value: stats.total_tokens_generated || 0 },
                { label: 'Avg Tokens/Second', value: (stats.avg_tokens_per_second || 0).toFixed(1) },
                { label: 'Uptime', value: this.formatDuration(stats.uptime_seconds) },
                { label: 'Device', value: stats.device },
                { label: 'Model Status', value: stats.model_loaded ? 'Loaded' : 'Not Loaded' }
            ];
            
            if (stats.cache_memory) {
                statItems.push({
                    label: 'Cache Memory',
                    value: stats.cache_memory.total_memory_mb.toFixed(1) + ' MB'
                });
            }
            
            statItems.forEach(item => {
                const statDiv = document.createElement('div');
                statDiv.className = 'stat-item';
                statDiv.innerHTML = `
                    <div class="stat-label">${item.label}</div>
                    <div class="stat-value">${item.value}</div>
                `;
                statsContent.appendChild(statDiv);
            });
            
            document.getElementById('statsModal').style.display = 'block';
            
        } catch (error) {
            this.showError('Failed to load statistics: ' + error.message);
        }
    }
    
    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    async exportConversation() {
        try {
            const response = await fetch('/api/export_conversation');
            const data = await response.json();
            
            if (response.ok) {
                // Download as JSON file
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `conversation_${data.timestamp.replace(/[:.]/g, '-')}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.showSuccess('Conversation exported!');
            } else {
                this.showError(data.error || 'Failed to export conversation');
            }
        } catch (error) {
            this.showError('Failed to export conversation: ' + error.message);
        }
    }
    
    async clearConversation() {
        if (!confirm('Are you sure you want to clear the conversation history?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/clear_conversation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                // Clear UI messages (except welcome)
                const messages = this.chatMessages.querySelectorAll('.message:not(.welcome-message .message)');
                messages.forEach(msg => msg.remove());
                
                this.showSuccess('Conversation cleared!');
                this.updateStatus('Ready');
            } else {
                this.showError('Failed to clear conversation');
            }
        } catch (error) {
            this.showError('Failed to clear conversation: ' + error.message);
        }
    }
    
    showSuccess(message) {
        // Simple success notification
        const notification = document.createElement('div');
        notification.className = 'notification success';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4caf50;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1001;
            animation: slideIn 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Modal Management
function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function saveConfig() {
    if (window.chatInterface) {
        window.chatInterface.saveConfig();
    }
}

function refreshStats() {
    if (window.chatInterface) {
        window.chatInterface.showStatsModal();
    }
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize chat interface when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.chatInterface = new ChatInterface();
});