/* Sidebar.js — Session List, Favorites, Admin Docs */

const Sidebar = {
    _currentSessionId: null,
    _adminDocsLoaded: false,

    async refresh() {
        await Promise.all([this.loadRecent(), this.loadFavorites()]);
    },

    clear() {
        document.getElementById('recentChats').innerHTML = '';
        document.getElementById('favChats').innerHTML = '';
        this._currentSessionId = null;
    },

    async loadRecent() {
        try {
            const resp = await fetch('/api/chat/sessions');
            if (!resp.ok) return;
            const sessions = await resp.json();
            this._renderList('recentChats', sessions, false);
        } catch (e) {
            console.error('loadRecent:', e);
        }
    },

    async loadFavorites() {
        try {
            const resp = await fetch('/api/chat/sessions?favorite_only=true');
            if (!resp.ok) return;
            const sessions = await resp.json();
            this._renderList('favChats', sessions, true);
        } catch (e) {
            console.error('loadFavorites:', e);
        }
    },

    _renderList(containerId, sessions, isFav) {
        const container = document.getElementById(containerId);
        if (sessions.length === 0) {
            container.innerHTML = `<div style="padding:4px 12px;font-size:12px;color:#666">${isFav ? '暂无收藏' : '暂无聊天'}</div>`;
            return;
        }
        let html = '';
        for (const s of sessions) {
            const active = s.id === this._currentSessionId ? ' active' : '';
            html += `<div class="sidebar-item${active}" data-sid="${s.id}" onclick="Sidebar.selectSession(${s.id})">
                <span class="item-title" title="${this._esc(s.title)}">${this._esc(s.title)}</span>
                <span class="star-btn${s.is_favorite ? ' favorited' : ''}"
                      onclick="event.stopPropagation();Sidebar.toggleFav(${s.id}, ${!s.is_favorite})"
                      title="${s.is_favorite ? '取消收藏' : '收藏'}">★</span>
                <span class="del-btn"
                      onclick="event.stopPropagation();Sidebar.deleteSession(${s.id})"
                      title="删除">✕</span>
            </div>`;
        }
        container.innerHTML = html;
    },

    async selectSession(sessionId) {
        this._currentSessionId = sessionId;
        this.refresh();
        document.getElementById('emptyState').style.display = 'none';
        document.getElementById('headerTitle').textContent = '加载中...';

        try {
            const resp = await fetch(`/api/chat/sessions/${sessionId}`);
            if (!resp.ok) throw new Error('Not found');
            const detail = await resp.json();
            document.getElementById('headerTitle').textContent = detail.title || '智旅云图';
            Chat.renderMessages(detail.messages);
        } catch (e) {
            console.error('selectSession:', e);
        }
    },

    async newSession() {
        this._currentSessionId = null;
        this.refresh();
        Chat.clear();
        document.getElementById('headerTitle').textContent = '智旅云图';
        document.getElementById('emptyState').style.display = 'flex';
        document.getElementById('msgInput').focus();
    },

    async toggleFav(sessionId, isFavorite) {
        try {
            await fetch(`/api/chat/sessions/${sessionId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_favorite: isFavorite }),
            });
            this.refresh();
        } catch (e) {
            console.error('toggleFav:', e);
        }
    },

    async deleteSession(sessionId) {
        if (!confirm('确定要删除这个聊天吗？')) return;
        try {
            await fetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
            if (this._currentSessionId === sessionId) {
                this._currentSessionId = null;
                Chat.clear();
                document.getElementById('headerTitle').textContent = '智旅云图';
                document.getElementById('emptyState').style.display = 'flex';
            }
            this.refresh();
        } catch (e) {
            console.error('deleteSession:', e);
        }
    },

    async loadAdminDocs() {
        if (this._adminDocsLoaded) return;
        try {
            const resp = await fetch('/api/admin/chunks');
            if (!resp.ok) {
                document.getElementById('adminHint').textContent = '权限不足';
                document.getElementById('adminHint').hidden = false;
                return;
            }
            const chunks = await resp.json();
            let html = '';
            for (const c of chunks.slice(0, 20)) {
                html += `<div class="admin-chunk-item">
                    <div class="chunk-title">${this._esc(c.title || '无标题')} (${this._esc(c.source || '')})</div>
                    <div class="chunk-text">${this._esc((c.text || '').substring(0, 150))}</div>
                </div>`;
            }
            document.getElementById('adminPanel').innerHTML = html;
            this._adminDocsLoaded = true;
        } catch (e) {
            console.error('loadAdminDocs:', e);
            document.getElementById('adminHint').textContent = '加载失败';
            document.getElementById('adminHint').hidden = false;
        }
    },

    _esc(s) {
        if (!s) return '';
        const div = document.createElement('div');
        div.textContent = s;
        return div.innerHTML;
    },
};
