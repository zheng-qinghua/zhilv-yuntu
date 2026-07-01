/* Auth.js — Login, Register, Session Management */

const Auth = {
    _mode: 'login',  // 'login' | 'register'
    _user: null,

    async init() {
        const resp = await fetch('/api/auth/me');
        if (resp.ok) {
            this._user = await resp.json();
            this.hideOverlay();
            this.updateUI();
        } else {
            this.showOverlay();
        }
        return this._user;
    },

    showOverlay() {
        document.getElementById('authOverlay').style.display = 'flex';
        document.getElementById('mainArea').style.opacity = '0.3';
    },

    hideOverlay() {
        document.getElementById('authOverlay').style.display = 'none';
        document.getElementById('mainArea').style.opacity = '1';
    },

    toggleMode() {
        this._mode = (this._mode === 'login') ? 'register' : 'login';
        document.getElementById('authTitle').textContent =
            this._mode === 'login' ? '登录 智旅云图' : '注册 智旅云图';
        document.getElementById('authSubmit').textContent =
            this._mode === 'login' ? '登 录' : '注 册';
        document.getElementById('authSwitchText').textContent =
            this._mode === 'login' ? '没有账号？' : '已有账号？';
        document.getElementById('authSwitchLink').textContent =
            this._mode === 'login' ? '立即注册' : '去登录';
        document.getElementById('authError').textContent = '';
    },

    async submit() {
        const username = document.getElementById('authUsername').value.trim();
        const password = document.getElementById('authPassword').value.trim();
        const errEl = document.getElementById('authError');
        errEl.textContent = '';

        if (!username || !password) {
            errEl.textContent = '请输入用户名和密码';
            return;
        }

        const btn = document.getElementById('authSubmit');
        btn.disabled = true;
        btn.textContent = '处理中...';

        const url = this._mode === 'login' ? '/api/auth/login' : '/api/auth/register';
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            if (resp.ok) {
                this._user = await resp.json();
                this.hideOverlay();
                this.updateUI();
                document.getElementById('authUsername').value = '';
                document.getElementById('authPassword').value = '';
            } else {
                const data = await resp.json();
                errEl.textContent = data.detail || '操作失败';
            }
        } catch (e) {
            errEl.textContent = '网络错误，请检查服务器连接';
        }
        btn.disabled = false;
        btn.textContent = this._mode === 'login' ? '登 录' : '注 册';
    },

    async logout() {
        await fetch('/api/auth/logout', { method: 'POST' });
        this._user = null;
        document.getElementById('sidebarUsername').textContent = '';
        this.showOverlay();
        Sidebar.clear();
        Chat.clear();
    },

    updateUI() {
        if (!this._user) return;
        document.getElementById('sidebarUsername').textContent = this._user.username;
        if (this._user.is_admin) {
            // Auto-load and show admin docs on login
            document.getElementById('adminPanel').style.display = 'block';
            Sidebar.loadAdminDocs();
        } else {
            document.getElementById('adminPanel').style.display = 'none';
            document.getElementById('adminHint').hidden = true;
        }
        Sidebar.refresh();
    },

    isAdmin() { return this._user && this._user.is_admin; },
    getUser() { return this._user; },
};
