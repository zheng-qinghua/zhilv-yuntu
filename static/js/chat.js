/* Chat.js — Message Rendering, Markdown, Send */

const Chat = {
    _rendering: false,

    clear() {
        const container = document.getElementById('chatMessages');
        container.innerHTML = `<div class="empty-state" id="emptyState">
            <div class="empty-logo">智旅</div>
            <h2>今天想去哪里旅行？</h2>
            <p>我可以帮你规划行程，也可以回答任何旅行相关问题</p>
            <div class="example-chips">
                <span class="example-chip" data-text="大理有什么值得去的景点？">大理有什么值得去的景点？</span>
                <span class="example-chip" data-text="帮我规划一个成都3天行程，预算3000">帮我规划一个成都3天行程，预算3000</span>
                <span class="example-chip" data-text="推荐几个西安必吃的美食">推荐几个西安必吃的美食</span>
                <span class="example-chip" data-text="三亚适合几月份去？有什么要注意的？">三亚适合几月份去？有什么要注意的？</span>
            </div>
        </div>`;
        this._bindChips();
    },

    renderMessages(messages) {
        if (!messages || messages.length === 0) {
            this.clear();
            return;
        }
        const container = document.getElementById('chatMessages');
        let html = '<div class="msg-wrapper">';
        for (const m of messages) {
            html += this._renderBubble(m);
        }
        html += '</div>';
        container.innerHTML = html;
        container.scrollTop = container.scrollHeight;

        // bind RAG source toggles
        container.querySelectorAll('.rag-sources-toggle').forEach(btn => {
            btn.addEventListener('click', function () {
                const list = this.nextElementSibling;
                if (list) {
                    list.classList.toggle('open');
                    const count = list.querySelectorAll('.rag-source-item').length;
                    this.textContent = list.classList.contains('open')
                        ? `收起参考来源 (${count}) ▲`
                        : `查看参考来源 (${count}) ▼`;
                }
            });
        });
    },

    appendAssistantBubble(content, ragSources) {
        if (this._rendering) return;
        const container = document.getElementById('chatMessages');
        // remove empty state if present
        const empty = document.getElementById('emptyState');
        if (empty) empty.style.display = 'none';

        // remove loading placeholder
        const loading = document.getElementById('loadingBubble');
        if (loading) loading.remove();

        // ensure wrapper exists
        let wrapper = container.querySelector('.msg-wrapper');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.className = 'msg-wrapper';
            container.appendChild(wrapper);
        }

        const row = document.createElement('div');
        row.className = 'msg-row assistant';
        row.innerHTML = this._renderBubble({
            role: 'assistant',
            content: content,
            rag_sources: ragSources,
        });
        wrapper.appendChild(row);
        container.scrollTop = container.scrollHeight;

        // bind RAG toggle
        row.querySelectorAll('.rag-sources-toggle').forEach(btn => {
            btn.addEventListener('click', function () {
                const list = this.nextElementSibling;
                if (list) {
                    list.classList.toggle('open');
                    const count = list.querySelectorAll('.rag-source-item').length;
                    this.textContent = list.classList.contains('open')
                        ? `收起参考来源 (${count}) ▲`
                        : `查看参考来源 (${count}) ▼`;
                }
            });
        });
    },

    appendUserBubble(content, imageBlobUrl) {
        const container = document.getElementById('chatMessages');
        const empty = document.getElementById('emptyState');
        if (empty) empty.style.display = 'none';

        let wrapper = container.querySelector('.msg-wrapper');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.className = 'msg-wrapper';
            container.appendChild(wrapper);
        }

        const row = document.createElement('div');
        row.className = 'msg-row user';
        row.innerHTML = this._renderBubble({ role: 'user', content: content, _imageBlobUrl: imageBlobUrl });
        wrapper.appendChild(row);
        container.scrollTop = container.scrollHeight;
    },

    showLoading() {
        const container = document.getElementById('chatMessages');
        const empty = document.getElementById('emptyState');
        if (empty) empty.style.display = 'none';

        let wrapper = container.querySelector('.msg-wrapper');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.className = 'msg-wrapper';
            container.appendChild(wrapper);
        }

        const row = document.createElement('div');
        row.className = 'msg-row assistant';
        row.id = 'loadingBubble';
        row.innerHTML = `<div class="msg-avatar">智</div>
            <div class="msg-bubble">
                <div class="loading-dots"><span></span><span></span><span></span></div>
            </div>`;
        wrapper.appendChild(row);
        container.scrollTop = container.scrollHeight;
    },

    _renderBubble(m) {
        const isUser = m.role === 'user';
        const avatarChar = isUser ? '我' : '智';
        const avatar = `<div class="msg-avatar">${avatarChar}</div>`;

        // Build image HTML if present
        let imageHtml = '';
        if (isUser && m._imageBlobUrl) {
            imageHtml = `<div class="msg-image"><img src="${m._imageBlobUrl}" alt="uploaded image"></div>`;
        }
        // Detect ![](url) markdown in content (for history loading in user messages)
        const imgMdMatch = (m.content || '').match(/^!\[\]\(\/photo\/([^)]+)\)/);
        if (imgMdMatch && isUser && !m._imageBlobUrl) {
            imageHtml = `<div class="msg-image"><img src="/photo/${imgMdMatch[1]}" alt="uploaded image"></div>`;
        }

        // Parse content — strip image markdown prefix for user messages with images
        let contentToRender = m.content || '';
        if (isUser && imgMdMatch) {
            contentToRender = contentToRender.replace(/^!\[\]\(\/photo\/[^)]+\)\n*\n*/, '');
        }
        const content = this._parseMarkdown(contentToRender);
        let bubble = `<div class="msg-bubble">${imageHtml}${content}</div>`;

        // RAG sources for assistant
        let sourcesHtml = '';
        if (!isUser && m.rag_sources && Array.isArray(m.rag_sources) && m.rag_sources.length > 0) {
            sourcesHtml = '<div class="rag-sources">';
            sourcesHtml += `<button class="rag-sources-toggle">查看参考来源 (${m.rag_sources.length}) ▼</button>`;
            sourcesHtml += '<div class="rag-sources-list">';
            for (const s of m.rag_sources) {
                sourcesHtml += `<div class="rag-source-item">${this._esc(s)}</div>`;
            }
            sourcesHtml += '</div></div>';
        }

        const bubbleBlock = `<div class="msg-bubble">${imageHtml}${content}${sourcesHtml}</div>`;

        if (isUser) {
            return `${bubbleBlock}${avatar}`;
        }
        return `${avatar}${bubbleBlock}`;
    },

    _parseMarkdown(text) {
        if (!text) return '';
        let html = text;

        // code blocks (triple backtick)
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g,
            '<pre><code>$2</code></pre>');

        // inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

        // horizontal rule
        html = html.replace(/^---$/gm, '<hr>');

        // list items
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        // wrap consecutive li's
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

        // paragraphs: double newlines
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';
        // clean empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');

        return html;
    },

    _esc(s) {
        if (!s) return '';
        const div = document.createElement('div');
        div.textContent = s;
        return div.innerHTML;
    },

    _bindChips() {
        document.querySelectorAll('.example-chip').forEach(chip => {
            chip.addEventListener('click', function () {
                const text = this.getAttribute('data-text');
                if (text) {
                    document.getElementById('msgInput').value = text;
                    App.send();
                }
            });
        });
    },
};
