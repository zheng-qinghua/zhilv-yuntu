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
                document.getElementById('adminHint').textContent = '需要管理员权限才能查看参考文档';
                document.getElementById('adminHint').hidden = false;
                return;
            }
            const chunks = await resp.json();
            this._renderAdminChunks(chunks);
            this._adminDocsLoaded = true;
        } catch (e) {
            console.error('loadAdminDocs:', e);
            document.getElementById('adminHint').textContent = '加载失败，请检查网络连接';
            document.getElementById('adminHint').hidden = false;
        }
    },

    _renderAdminChunks(chunks) {
        let html = '';
        this._adminChunksCache = {};
        for (const c of chunks.slice(0, 20)) {
            this._adminChunksCache[c.id] = c;
            const escapedTitle = this._esc(c.title || '无标题');
            const escapedSource = this._esc(c.source || '');
            const escapedText = this._esc((c.text || '').substring(0, 150));
            html += `<div class="admin-chunk-item" data-chunk-id="${this._esc(c.id)}" onclick="Sidebar.viewChunk('${this._esc(c.id)}')" title="点击查看全文">
                <div class="chunk-title">${escapedTitle} (${escapedSource})</div>
                <div class="chunk-text">${escapedText}...</div>
            </div>`;
        }
        document.getElementById('adminPanel').innerHTML = html;
    },

    async viewChunk(chunkId) {
        const modal = document.getElementById('docModal');
        document.getElementById('docModalTitle').textContent = '加载中...';
        document.getElementById('docModalBody').innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
        modal.style.display = 'flex';

        try {
            const resp = await fetch(`/api/admin/chunks/${encodeURIComponent(chunkId)}`);
            if (!resp.ok) {
                document.getElementById('docModalBody').innerHTML = '<p style="color:#e74c3c">加载失败：权限不足或文档不存在</p>';
                return;
            }
            const chunk = await resp.json();
            document.getElementById('docModalTitle').textContent = `${chunk.title || '无标题'} (${chunk.source || '未知来源'})`;
            document.getElementById('docModalBody').innerHTML = this._esc(chunk.text || '无内容');
        } catch (e) {
            console.error('viewChunk:', e);
            document.getElementById('docModalBody').innerHTML = '<p style="color:#e74c3c">加载失败：网络错误</p>';
        }
    },

    closeDocModal() {
        document.getElementById('docModal').style.display = 'none';
    },

    _initDocModal() {
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                const modal = document.getElementById('docModal');
                if (modal && modal.style.display === 'flex') {
                    modal.style.display = 'none';
                }
            }
        });
    },

    async toggleAdminDocs() {
        const panel = document.getElementById('adminPanel');
        const arrow = document.getElementById('refDocsArrow');
        const hint = document.getElementById('adminHint');

        if (panel.style.display === 'block') {
            panel.style.display = 'none';
            if (arrow) arrow.innerHTML = '&#9660;';
            hint.hidden = true;
        } else {
            await this.loadAdminDocs();
            if (this._adminDocsLoaded) {
                panel.style.display = 'block';
                if (arrow) arrow.innerHTML = '&#9650;';
                hint.hidden = true;
            }
        }
    },

    _esc(s) {
        if (!s) return '';
        const div = document.createElement('div');
        div.textContent = s;
        return div.innerHTML;
    },
};
