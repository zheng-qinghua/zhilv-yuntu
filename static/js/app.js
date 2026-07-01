/* App.js — Main Controller */

const App = {
    _sending: false,
    _imageFile: null,
    _imageBlobUrl: null,

    _bindEvents() {
        // Bind UI event handlers (always, regardless of auth state)
        document.getElementById('btnNewChat').onclick = () => Sidebar.newSession();
        document.getElementById('btnLogout').onclick = () => Auth.logout();

        // Init document modal escape key
        Sidebar._initDocModal();

        document.querySelectorAll('.example-chip').forEach(chip => {
            chip.addEventListener('click', function () {
                const text = this.getAttribute('data-text');
                if (text) {
                    document.getElementById('msgInput').value = text;
                    App.send();
                }
            });
        });

        const textarea = document.getElementById('msgInput');
        textarea.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 160) + 'px';
        });
    },

    async init() {
        // Always bind events first
        this._bindEvents();

        const user = await Auth.init();
        if (!user) return;

        Sidebar.refresh();
        console.log('[App] 智旅云图 初始化完成');
    },

    handleImageSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            alert('请选择图片文件');
            return;
        }

        this._imageFile = file;
        // Revoke previous blob URL
        if (this._imageBlobUrl) URL.revokeObjectURL(this._imageBlobUrl);
        this._imageBlobUrl = URL.createObjectURL(file);

        // Show preview below input
        document.getElementById('imagePreviewImg').src = this._imageBlobUrl;
        document.getElementById('imagePreview').style.display = 'inline-block';
        document.getElementById('btnImageUpload').classList.add('has-image');
    },

    clearImage() {
        this._imageFile = null;
        if (this._imageBlobUrl) {
            URL.revokeObjectURL(this._imageBlobUrl);
            this._imageBlobUrl = null;
        }
        document.getElementById('imageFileInput').value = '';
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('imagePreviewImg').src = '';
        document.getElementById('btnImageUpload').classList.remove('has-image');
    },

    async send() {
        if (this._sending) return;

        const textarea = document.getElementById('msgInput');
        const content = textarea.value.trim();

        // Must have either text or image
        if (!content && !this._imageFile) return;

        const btn = document.getElementById('btnSend');
        this._sending = true;
        btn.disabled = true;
        btn.classList.add('loading');

        // Show user message with image (if any)
        Chat.appendUserBubble(content, this._imageBlobUrl);
        textarea.value = '';
        textarea.style.height = 'auto';

        Chat.showLoading();

        try {
            let resp;
            if (this._imageFile) {
                // Image upload endpoint
                const formData = new FormData();
                formData.append('file', this._imageFile);
                formData.append('session_id', Sidebar._currentSessionId || '');
                formData.append('message', content);

                resp = await fetch('/api/chat/upload-image', {
                    method: 'POST',
                    body: formData,
                });
            } else {
                // Text-only endpoint
                resp = await fetch('/api/chat/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: Sidebar._currentSessionId,
                        content: content,
                    }),
                });
            }

            if (!resp.ok) {
                const err = await resp.json();
                document.getElementById('loadingBubble')?.remove();
                Chat.appendAssistantBubble('Error: ' + (err.detail || 'Unknown error'), []);
            } else {
                const data = await resp.json();
                if (!Sidebar._currentSessionId) {
                    Sidebar._currentSessionId = data.session_id;
                    const title = this._imageFile
                        ? '图片: ' + this._imageFile.name
                        : content.substring(0, 30) + (content.length > 30 ? '...' : '');
                    document.getElementById('headerTitle').textContent = title;
                }
                document.getElementById('loadingBubble')?.remove();
                Chat.appendAssistantBubble(data.content, data.rag_sources);
                Sidebar.refresh();
            }
        } catch (e) {
            document.getElementById('loadingBubble')?.remove();
            Chat.appendAssistantBubble('Network error: ' + e.message, []);
        }

        // Clear image state
        this.clearImage();

        this._sending = false;
        btn.disabled = false;
        btn.classList.remove('loading');
        textarea.focus();
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
