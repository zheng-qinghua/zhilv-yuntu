/* App.js — Main Controller */

const App = {
    _sending: false,

    async init() {
        // Auth check
        const user = await Auth.init();
        if (!user) return;

        // Event listeners
        document.getElementById('btnNewChat').onclick = () => Sidebar.newSession();
        document.getElementById('btnLogout').onclick = () => Auth.logout();

        // Example chips
        document.querySelectorAll('.example-chip').forEach(chip => {
            chip.addEventListener('click', function () {
                const text = this.getAttribute('data-text');
                if (text) {
                    document.getElementById('msgInput').value = text;
                    App.send();
                }
            });
        });

        // Auto-resize textarea
        const textarea = document.getElementById('msgInput');
        textarea.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 160) + 'px';
        });

        // Load initial sessions
        Sidebar.refresh();

        console.log('[App] 智旅云图 初始化完成');
    },

    async send() {
        if (this._sending) return;

        const textarea = document.getElementById('msgInput');
        const content = textarea.value.trim();
        if (!content) return;

        const btn = document.getElementById('btnSend');
        this._sending = true;
        btn.disabled = true;
        btn.classList.add('loading');

        // Show user message
        Chat.appendUserBubble(content);
        textarea.value = '';
        textarea.style.height = 'auto';

        // Show loading
        Chat.showLoading();

        try {
            const resp = await fetch('/api/chat/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: Sidebar._currentSessionId,
                    content: content,
                }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                document.getElementById('loadingBubble')?.remove();
                Chat.appendAssistantBubble(`❌ 出错了: ${err.detail || '未知错误'}`, []);
            } else {
                const data = await resp.json();
                if (!Sidebar._currentSessionId) {
                    Sidebar._currentSessionId = data.session_id;
                    document.getElementById('headerTitle').textContent =
                        content.substring(0, 30) + (content.length > 30 ? '...' : '');
                }
                document.getElementById('loadingBubble')?.remove();
                Chat.appendAssistantBubble(data.content, data.rag_sources);
                Sidebar.refresh();
            }
        } catch (e) {
            document.getElementById('loadingBubble')?.remove();
            Chat.appendAssistantBubble(`❌ 网络错误: ${e.message}`, []);
        }

        this._sending = false;
        btn.disabled = false;
        btn.classList.remove('loading');
        textarea.focus();
    },
};

// Startup
document.addEventListener('DOMContentLoaded', () => App.init());
