# LangChain + RAG + Agent 完全学习指南

> 从零基础到实战项目，全面掌握 LangChain、RAG（检索增强生成）和 AI Agent 三大核心技术。
> 更新日期：2026 年 5 月

---

## 目录

1. [环境准备](#1-环境准备)
2. [LangChain 核心概念](#2-langchain-核心概念)
3. [RAG 检索增强生成](#3-rag-检索增强生成)
4. [AI Agent 智能代理](#4-ai-agent-智能代理)
5. [实战项目：智能知识库问答系统](#5-实战项目智能知识库问答系统)
6. [进阶技巧](#6-进阶技巧)
7. [常见问题与排错](#7-常见问题与排错)

---

## 1. 环境准备

### 1.1 安装 Python 依赖

```bash
# 核心框架
pip install langchain langchain-core langchain-openai langchain-community

# RAG 相关
pip install langchain-text-splitters chromadb faiss-cpu
pip install pypdf pymupdf sentence-transformers

# Agent 相关
pip install langgraph

# 前端（可选）
pip install streamlit gradio

# 环境变量管理
pip install python-dotenv
```

### 1.2 配置 API Key

创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1   # 或用国内代理

# 如果用 DeepSeek 等国产模型
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

```python
# config.py - 配置加载
import os
from dotenv import load_dotenv

load_dotenv()

# 模型配置
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# 向量数据库路径
VECTOR_STORE_PATH = "./data/chroma_db"
```

---

## 2. LangChain 核心概念

### 2.1 什么是 LangChain？

LangChain 是一个用于构建 LLM（大语言模型）驱动应用的框架。它提供了标准化的接口来连接模型、数据源、工具和外部系统。

**核心设计理念：组件化 + 可组合**

```
┌─────────────────────────────────────────────────┐
│                  LangChain 架构                    │
├─────────────┬──────────────┬─────────────────────┤
│   Models    │   Retrieval  │      Agents         │
│  模型层     │   检索层      │     代理层          │
├─────────────┼──────────────┼─────────────────────┤
│  LLM/Chat   │  VectorStore │  Tools 工具          │
│  Embeddings │  Retriever   │  AgentExecutor      │
│  Prompt     │  Document    │  Memory 记忆         │
└─────────────┴──────────────┴─────────────────────┘
```

### 2.2 模型 (Models)

LangChain 支持多种模型提供商，通过统一接口调用。

#### 2.2.1 基础 LLM 调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 初始化聊天模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",      # 模型名称
    temperature=0.7,             # 温度：控制随机性 (0=确定, 1=创意)
    max_tokens=1024,             # 最大输出长度
    openai_api_key="sk-xxx",    # API Key（也可通过环境变量）
    openai_api_base="https://api.openai.com/v1"  # API 地址
)

# 基础调用
response = llm.invoke("你好，请介绍一下自己")
print(response.content)
```

#### 2.2.2 多轮对话

```python
messages = [
    SystemMessage(content="你是一个有用的助手，请用中文回答。"),
    HumanMessage(content="我叫小明"),
    AIMessage(content="你好小明！有什么可以帮助你的吗？"),
    HumanMessage(content="我叫什么名字？"),
]

response = llm.invoke(messages)
print(response.content)  # 输出：你叫小明
```

#### 2.2.3 使用不同模型提供商

```python
# DeepSeek 模型
from langchain_openai import ChatOpenAI

deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-xxx",
    openai_api_base="https://api.deepseek.com"
)

# Ollama 本地模型
from langchain_ollama import ChatOllama

ollama_llm = ChatOllama(model="llama3:8b")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic

claude_llm = ChatAnthropic(model="claude-sonnet-4-6")
```

#### 2.2.4 流式输出

```python
# 流式输出：逐字返回，提升用户体验
for chunk in llm.stream("写一首关于夏天的诗"):
    print(chunk.content, end="", flush=True)
```

### 2.3 提示词模板 (Prompt Templates)

提示词模板让你可以用变量动态构建 prompt。

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder

# ========== 基础模板 ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请用{language}回答。"),
    ("human", "{question}")
])

# 传入变量，生成最终 prompt
formatted = prompt.invoke({
    "role": "Python 编程专家",
    "language": "中文",
    "question": "如何在 Python 中处理 JSON？"
})

print(formatted.to_string())
# 输出：
# System: 你是一个Python 编程专家，请用中文回答。
# Human: 如何在 Python 中处理 JSON？

# ========== 带对话历史的模板 ==========
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
    ("human", "{input}")
])

# 使用方式
response = llm.invoke(chat_prompt.invoke({
    "history": [HumanMessage(content="你好"), AIMessage(content="你好！")],
    "input": "我刚才说了什么？"
}))
```

### 2.4 LCEL 链式表达式（重要！）

LCEL（LangChain Expression Language）是 LangChain 的核心编程范式，使用 `|` 管道符像搭积木一样组合组件。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ========== LCEL 基础链 ==========
prompt = ChatPromptTemplate.from_template("给{product}写一句广告语")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 用管道符串联：prompt → llm → 输出解析器
chain = prompt | llm | StrOutputParser()

# 调用链
result = chain.invoke({"product": "智能手表"})
print(result)
# 输出：掌控时间，智领生活——xx智能手表

# ========== LCEL 流式输出 ==========
for chunk in chain.stream({"product": "智能手表"}):
    print(chunk, end="", flush=True)

# ========== LCEL 批量处理 ==========
inputs = [{"product": "耳机"}, {"product": "键盘"}, {"product": "鼠标"}]
results = chain.batch(inputs)
```

**旧的写法 vs 新的 LCEL 写法（对比）：**

```python
# ❌ 旧写法（已废弃，不推荐）
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("hello")

# ✅ 新写法（LCEL，推荐）
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "hello"})
```

### 2.5 输出解析器 (Output Parsers)

#### 2.5.1 结构化输出（Pydantic）

这是最有用的特性——让 LLM 输出格式化的 JSON 数据。

```python
from pydantic import BaseModel, Field
from typing import List

# 定义输出结构
class PersonInfo(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    skills: List[str] = Field(description="技能列表")
    summary: str = Field(description="一句话总结")

# 使用结构化输出
structured_llm = llm.with_structured_output(PersonInfo)

result = structured_llm.invoke("张三，25岁，会Python、Java和React，是个全栈工程师")
print(result.name)    # 张三
print(result.age)     # 25
print(result.skills)  # ['Python', 'Java', 'React']
print(result.summary) # 张三是一位25岁的全栈工程师
print(type(result))   # <class '__main__.PersonInfo'> ← 真正的 Pydantic 对象！
```

#### 2.5.2 字符串解析器

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

# 最简单的解析器——提取 content 字符串
parser = StrOutputParser()
chain = prompt | llm | parser
result = chain.invoke({"topic": "AI"})
# result 直接是字符串，不需要 .content

# JSON 解析器
json_parser = JsonOutputParser()
chain = prompt | llm | json_parser
result = chain.invoke({"topic": "AI"})
# result 是 Python 字典
```

### 2.6 记忆系统 (Memory)

让 LLM 记住之前的对话内容。

```python
from langchain_core.messages import HumanMessage, AIMessage

# ========== 手动管理历史（推荐） ==========
history = []

def chat(user_input: str) -> str:
    """简单的手动对话记忆管理"""
    messages = [("system", "你是乐于助人的助手")] + history + [("human", user_input)]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    
    # 保存到历史
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
    return response

# ========== 使用 RunnableWithMessageHistory ==========
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 存储各会话的历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用
response = chain_with_memory.invoke(
    {"input": "我叫小明"},
    config={"configurable": {"session_id": "user-001"}}
)
# 第二次调用会记住之前的内容
response2 = chain_with_memory.invoke(
    {"input": "我叫什么？"},
    config={"configurable": {"session_id": "user-001"}}
)
# 输出：你叫小明
```

---

## 3. RAG 检索增强生成

### 3.1 什么是 RAG？

RAG（Retrieval-Augmented Generation）是一种让 LLM 在回答问题时先检索外部知识库中相关文档，再基于检索结果生成答案的技术。

**为什么需要 RAG？**
- LLM 的训练数据有截止日期，不知道最新信息
- LLM 可能产生幻觉（编造不存在的事实）
- 企业有自己的私有文档需要 LLM 基于它们回答

**RAG 的工作流程：**

```
用户提问
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. 加载     │ ──▶ │  2. 切分     │ ──▶ │  3. 向量化   │
│  文档       │     │  文档块     │     │  Embedding  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  6. 生成回答 │ ◀── │  5. 构建提示  │ ◀── │  4. 存储到    │
│  (LLM)     │     │  Context    │     │  向量数据库  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 3.2 步骤一：加载文档

```python
from langchain_community.document_loaders import (
    PyPDFLoader,        # PDF 文件
    TextLoader,          # 纯文本
    CSVLoader,           # CSV 表格
    UnstructuredMarkdownLoader,  # Markdown
    WebBaseLoader,       # 网页
    DirectoryLoader,     # 批量加载目录
)

# ===== 加载 PDF =====
loader = PyPDFLoader("./docs/report.pdf")
pages = loader.load()
print(f"共加载 {len(pages)} 页")
print(f"第一页内容：{pages[0].page_content[:200]}...")

# ===== 加载网页 =====
loader = WebBaseLoader("https://docs.langchain.com")
web_docs = loader.load()

# ===== 批量加载目录下所有文件 =====
loader = DirectoryLoader(
    "./docs/",
    glob="**/*.pdf",         # 匹配所有 PDF
    loader_cls=PyPDFLoader,
    show_progress=True       # 显示进度条
)
all_docs = loader.load()
print(f"共加载 {len(all_docs)} 个文档")

# ===== 更精细的 PDF 加载（推荐） =====
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./docs/report.pdf")
documents = loader.load()
# PyMuPDF 更快，且能更好地处理复杂 PDF
```

### 3.3 步骤二：文本切分

切分的艺术在于：**块太小**丢失上下文，**块太大**影响检索精度。

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # 最常用，智能递归切分
    CharacterTextSplitter,           # 按字符切分
    TokenTextSplitter,               # 按 token 数切分
)

# ===== 最推荐的切分方式 =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 每块最大字符数
    chunk_overlap=200,     # 块之间重叠字符数（保留上下文）
    length_function=len,   # 计算长度的函数
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 优先级递减的分隔符
)

chunks = text_splitter.split_documents(documents)
print(f"切分为 {len(chunks)} 个文档块")

# ===== 查看切分结果 =====
for i, chunk in enumerate(chunks[:3]):
    print(f"=== 块 {i+1} (源：{chunk.metadata.get('source', 'unknown')}) ===")
    print(chunk.page_content[:200])
    print()

# ===== 不同场景的参数建议 =====
# 短问答：chunk_size=500, overlap=100
# 一般问答：chunk_size=1000, overlap=200
# 长文档摘要：chunk_size=2000, overlap=400
# 代码搜索：chunk_size=1500, overlap=300
```

### 3.4 步骤三：向量化（Embedding）

将文本转换成数字向量，语义相似的文本在向量空间中距离更近。

```python
# ===== 方案 A：OpenAI Embedding（云端，效果好） =====
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 性价比高
    # model="text-embedding-3-large",  # 效果最好，贵一些
)

# ===== 方案 B：本地 HuggingFace Embedding（免费，无需联网） =====
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文推荐
    model_kwargs={"device": "cpu"},        # 或 "cuda"
    encode_kwargs={"normalize_embeddings": True}
)

# ===== 方案 C：Ollama 本地 Embedding（最简单本地方案） =====
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ===== 测试向量化 =====
sample_text = "你好，这是一个测试文本"
vector = embeddings.embed_query(sample_text)
print(f"向量维度：{len(vector)}")  # 通常 768 或 1536 维
print(f"前 5 个值：{vector[:5]}")
```

### 3.5 步骤四：向量数据库存储

#### 方案 A：ChromaDB（推荐用于学习和中小项目）

```python
from langchain_chroma import Chroma

# ===== 创建向量数据库 =====
vectorstore = Chroma.from_documents(
    documents=chunks,               # 文档块列表
    embedding=embeddings,           # 向量化模型
    persist_directory="./chroma_db"  # 持久化目录
)
# ChromaDB 会自动保存，下次直接用

# ===== 持久化后重新加载 =====
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# ===== ChromaDB 的增删改查 =====
# 添加新文档
new_chunks = text_splitter.split_documents(new_documents)
vectorstore.add_documents(new_chunks)

# 通过 metadata 过滤删除
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])

# 更新文档
vectorstore.update_document(document_id="doc_id_1", document=updated_chunk)

# 查看存储信息
print(f"存储了 {vectorstore._collection.count()} 个文档块")
```

#### 方案 B：FAISS（高性能，适合只读场景）

```python
from langchain_community.vectorstores import FAISS

# ===== 创建 FAISS 数据库 =====
vectorstore = FAISS.from_documents(chunks, embeddings)

# ===== 保存到磁盘 =====
vectorstore.save_local("./faiss_db")

# ===== 从磁盘加载 =====
vectorstore = FAISS.load_local(
    "./faiss_db",
    embeddings,
    allow_dangerous_deserialization=True  # 安全起见需显式确认
)
```

**ChromaDB vs FAISS 对比：**

| 特性 | ChromaDB | FAISS |
|------|----------|-------|
| 增删改查 | 支持在线 CRUD | 不支持（需重建） |
| 持久化 | 自动持久化 | 手动 save_local |
| 性能 | 适合百万级 | 适合亿级（GPU加速） |
| 易用性 | 更简单 | 略复杂 |
| 适用场景 | 开发/中小项目 | 大规模生产环境 |

### 3.6 步骤五：检索与问答

#### 3.6.1 基础 RAG 链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 语义相似度检索
    search_kwargs={"k": 4}     # 返回最相似的 4 个文档
)

# 定义 RAG 提示词模板
rag_prompt = ChatPromptTemplate.from_template("""你是一个知识库问答助手。请根据以下检索到的上下文来回答问题。
如果上下文中没有相关信息，请如实说"根据已有资料无法回答"。

【检索到的上下文】
{context}

【用户问题】
{question}

【回答】""")

# 辅助函数：把文档列表格式化为文本
def format_docs(docs):
    """将检索到的文档格式化为上下文字符串"""
    return "\n\n---\n\n".join(
        f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )

# ===== 构建 RAG 链（LCEL 管道式） =====
rag_chain = (
    {
        "context": retriever | format_docs,   # 检索 → 格式化
        "question": RunnablePassthrough()     # 直接传递问题
    }
    | rag_prompt      # 填入提示词模板
    | llm             # 调用大模型
    | StrOutputParser()  # 解析输出
)

# ===== 测试问答 =====
question = "公司去年的营收是多少？"
answer = rag_chain.invoke(question)
print(answer)
```

#### 3.6.2 带来源标注的 RAG

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def format_docs_with_source(docs):
    """格式化文档并保留来源信息"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"[文档{i} - 来源:{source}, 页码:{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# 增强版提示词，要求引用来源
rag_prompt_with_citation = ChatPromptTemplate.from_template("""你是一个知识库问答助手。请根据以下检索到的上下文来回答问题。
如果上下文中没有相关信息，请如实说"根据已有资料无法回答"。
请在回答中使用 [文档X] 标注信息来源。

【检索到的上下文】
{context}

【用户问题】
{question}

【回答（请标注引用来源）】""")

rag_chain_with_source = (
    {
        "context": retriever | format_docs_with_source,
        "question": RunnablePassthrough()
    }
    | rag_prompt_with_citation
    | llm
    | StrOutputParser()
)

result = rag_chain_with_source.invoke("公司的主要产品是什么？")
print(result)
# 输出中包含 [文档1] 等引用标注
```

### 3.7 高级 RAG 技术

#### 3.7.1 混合检索（语义 + 关键词）

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 语义检索器
semantic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# BM25 关键词检索器（基于词频）
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 混合检索器——结合两者优势
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 语义权重 60%，关键词权重 40%
)

# 用混合检索器替换原来的检索器即可
```

#### 3.7.2 重排序（Reranking）

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 第一步：粗检索（获取较多候选）
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}  # 先取 20 个候选
)

# 第二步：用 Cross-Encoder 精排
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=5)  # 重排后取 top 5

# 压缩检索器 = 粗检索 + 精排
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 用精排后的检索器
rag_chain = (
    {
        "context": compression_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

#### 3.7.3 多轮对话 RAG

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 问题改写提示词：把多轮对话中的问题转为独立问题
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据聊天历史，将用户问题改写为独立、完整的问题。如果问题已经是独立的，则原样返回。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# 创建"历史感知"检索器
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# 最终的多轮 RAG 链
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题：\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain_multi_turn = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 使用，传入聊天历史
result = rag_chain_multi_turn.invoke({
    "input": "它的竞争对手是谁？",  # "它"会根据历史被解析
    "chat_history": [
        HumanMessage(content="苹果公司的主要产品是什么？"),
        AIMessage(content="苹果公司的主要产品包括iPhone、Mac、iPad...")
    ]
})
```

---

## 4. AI Agent 智能代理

### 4.1 什么是 Agent？

Agent 是能够自主使用工具来完成任务目标的 AI 系统。它不仅仅是回答问题，而是会**规划、推理、调用工具、观察结果、调整策略**。

```
┌─────────────────────────────────────────────────┐
│                 Agent 工作循环                     │
│                                                   │
│   用户输入 ──▶ 思考(Reason) ──▶ 行动(Act)        │
│                    ▲               │              │
│                    │               ▼              │
│                    └── 观察(Observe) ◀──┘         │
│                                                   │
│   循环直到：任务完成 / 达到最大步数 / 无法继续      │
└─────────────────────────────────────────────────┘
```

**Agent 的三大核心要素：**
1. **模型（Model）**：大脑，负责推理和决策
2. **工具（Tools）**：手脚，能执行具体操作
3. **编排器（Orchestrator）**：神经系统，协调模型和工具的交互

### 4.2 创建自定义工具

工具是 Agent 最核心的能力。@tool 装饰器是定义工具的最简单方式。

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ===== 基础工具：使用 @tool 装饰器 =====
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息。输入应为城市中文名称，如'北京'、'上海'。"""
    # 实际项目中这里调用天气 API
    weather_data = {
        "北京": "晴天，25°C，湿度45%",
        "上海": "多云，28°C，湿度65%",
        "广州": "阵雨，32°C，湿度80%",
    }
    return weather_data.get(city, f"未找到{city}的天气数据")

@tool
def calculator(expression: str) -> str:
    """执行数学计算。输入为一个数学表达式，如 '2 + 3 * 4'。"""
    try:
        # 安全评估（仅数学运算，不执行代码）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"

# ===== 高级工具：带 Pydantic 参数校验 =====
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="返回的最大结果数", ge=1, le=20)

@tool(args_schema=SearchInput)
def search_knowledge(query: str, max_results: int = 5) -> str:
    """在知识库中搜索相关信息。"""
    # 实际项目中调用搜索引擎或向量数据库
    results = [f"关于'{query}'的搜索结果 {i+1}" for i in range(max_results)]
    return "\n".join(results)

# ===== 异步工具 =====
@tool
async def fetch_url_content(url: str) -> str:
    """异步获取网页内容。"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()[:500]  # 截取前500字
```

### 4.3 创建 Agent

LangChain 提供了多种创建 Agent 的方式，推荐使用 `create_agent`（2025+ 新 API）或 `langgraph`。

#### 方式一：create_agent（推荐，最简单）

```python
from langchain.agents import create_agent

# 定义工具列表
tools = [get_weather, calculator, search_knowledge]

# 创建 Agent
agent = create_agent(
    model="gpt-3.5-turbo",   # 或用 ChatOpenAI 实例
    tools=tools,
    system_prompt="你是一个有用的助手，可以使用工具来回答用户的问题。请用中文回答。"
)

# 使用 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天天气怎么样？然后帮我计算 2的10次方"}]
})

# 提取最终回答
final_answer = result["messages"][-1].content
print(final_answer)
```

#### 方式二：LangGraph create_react_agent（更灵活）

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [get_weather, calculator, search_knowledge]

# 创建 ReAct Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    # 可以自定义 system prompt
    state_modifier="你是一个乐于助人的助手，请使用工具帮助用户解决问题。回答使用中文。"
)

# 运行 Agent
config = {"configurable": {"thread_id": "conversation-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "上海的天气怎样？"}]},
    config=config
)

# 获取最终回答
for msg in result["messages"]:
    if msg.type == "ai" and msg.content:
        print(msg.content)
```

#### 方式三：带记忆的 Agent

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 创建记忆管理器
memory = MemorySaver()

# 创建带记忆的 Agent
agent_with_memory = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,  # 关键：传入 checkpointer 实现记忆
)

# 使用同一个 thread_id 来保持对话
config = {"configurable": {"thread_id": "user-session-001"}}

# 第一次对话
r1 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "我叫小明"}]},
    config=config
)
print(r1["messages"][-1].content)  # 回复

# 第二次对话——Agent 会记住之前的对话
r2 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
print(r2["messages"][-1].content)  # 输出：你叫小明
```

### 4.4 理解 ReAct 模式

ReAct = **Reasoning（推理）+ Acting（行动）**，是 Agent 最核心的工作模式。

```
用户: 北京气温多少？然后算摄氏度转华氏度

Agent 思考: 我需要先查天气，然后做温度转换。
          → 行动: get_weather("北京")
          → 观察: "晴天，25°C"
Agent 思考: 已获取摄氏温度25度，现在计算华氏度。
          → 行动: calculator("25 * 9/5 + 32")
          → 观察: "计算结果：25 * 9/5 + 32 = 77.0"
Agent 思考: 得到华氏度77度，可以回答了。
          → 回答: "北京今天25°C（77°F），晴天。"
```

### 4.5 中间件（Middleware）

中间件可以在 Agent 执行的各个环节插入自定义逻辑。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,      # 模型故障转移
    SummarizationMiddleware,      # 长对话自动摘要
    HumanInTheLoopMiddleware,     # 人工审核
)

agent = create_agent(
    model="gpt-4o",
    tools=tools,
    middleware=[
        # 如果 gpt-4o 不可用，自动切换到备用模型
        ModelFallbackMiddleware(
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
        ),
        # 当对话历史过长时自动压缩
        SummarizationMiddleware(),
        # 执行敏感操作前需要人工确认
        HumanInTheLoopMiddleware(
            interrupt_on={"delete_record": True}  # delete_record 工具需要确认
        ),
    ],
)
```

---

## 5. 实战项目：智能知识库问答系统

### 5.1 项目概述

我们将构建一个完整的 **智能知识库问答系统**，融合三种技术：
- **LangChain**：框架基础，管理链和提示词
- **RAG**：检索增强，从 PDF 文档中获取知识
- **Agent**：智能代理，自主决定用哪种方式回答问题

**项目功能：**
1. 上传 PDF/文档 → 自动向量化存储
2. 用户可以提问 → RAG 检索相关文档
3. Agent 自主判断：用知识库还是用搜索引擎
4. 支持多轮对话，记住上下文
5. 回答带来源引用
6. 有 Web UI（Streamlit）

### 5.2 项目结构

```
knowledge_qa_system/
├── .env                    # 环境变量
├── requirements.txt        # 依赖列表
├── main.py                 # Streamlit 前端入口
├── config.py               # 配置管理
├── document_loader.py      # 文档加载与切分
├── vector_store.py         # 向量数据库管理
├── rag_chain.py            # RAG 检索链
├── tools.py                # Agent 工具定义
├── agent.py                # Agent 创建与管理
└── data/
    ├── uploads/            # 上传的文档
    └── chroma_db/          # 向量数据库存储
```

### 5.3 完整代码实现

#### 5.3.1 配置管理 `config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Embedding 配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 向量数据库路径
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/chroma_db")

# 文档配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# 上传文件配置
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
```

#### 5.3.2 文档加载与切分 `document_loader.py`

```python
import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR


def load_single_document(file_path: str) -> List[Document]:
    """加载单个文档"""
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式：{file_path}")

    documents = loader.load()

    # 为每个文档添加来源信息
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    return documents


def load_all_documents() -> List[Document]:
    """加载 data/uploads 下所有文档"""
    all_docs = []
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path) and filename.split(".")[-1] in ("pdf", "txt", "md"):
            try:
                docs = load_single_document(file_path)
                all_docs.extend(docs)
                print(f"✓ 已加载：{filename}（{len(docs)} 页/段）")
            except Exception as e:
                print(f"✗ 加载失败 {filename}：{e}")
    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """切分文档为小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def process_all_documents() -> List[Document]:
    """加载并切分所有文档（一步完成）"""
    documents = load_all_documents()
    if not documents:
        print("警告：未找到任何文档！请先把文件放到 data/uploads/ 目录")
        return []
    chunks = split_documents(documents)
    print(f"共 {len(documents)} 个文档 → 切分为 {len(chunks)} 个文本块")
    return chunks
```

#### 5.3.3 向量数据库管理 `vector_store.py`

```python
import os
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH


def get_embeddings():
    """获取嵌入模型"""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def create_vectorstore(documents: List[Document]) -> Chroma:
    """从文档创建向量数据库"""
    embeddings = get_embeddings()

    # 如果已有数据库，先删除
    import shutil
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
        os.makedirs(VECTOR_STORE_PATH)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )
    print(f"向量数据库已创建，存储 {len(documents)} 个文本块")
    return vectorstore


def load_vectorstore() -> Chroma | None:
    """加载已有的向量数据库"""
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        return None

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings,
    )
    count = vectorstore._collection.count()
    print(f"向量数据库已加载，包含 {count} 个文本块")
    return vectorstore


def add_to_vectorstore(vectorstore: Chroma, documents: List[Document]):
    """向已有向量数据库添加新文档"""
    vectorstore.add_documents(documents)
    print(f"添加了 {len(documents)} 个新文本块")


def get_retriever(vectorstore: Chroma, k: int = 4):
    """创建检索器"""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
```

#### 5.3.4 RAG 检索链 `rag_chain.py`

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List
from config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, RETRIEVAL_K


def format_docs(docs: List[Document]) -> str:
    """将检索文档格式化为上下文字符串"""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "N/A")
        formatted_parts.append(
            f"[文档{i} | 来源：{source}，页码：{page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_parts)


def create_rag_chain(retriever):
    """创建 RAG 问答链"""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.3,  # 低温度，保证准确性
    )

    prompt = ChatPromptTemplate.from_template("""你是一个知识库问答助手。请严格基于以下检索到的上下文来回答用户问题。

回答规则：
1. 如果上下文中有相关信息，请详细回答问题，并标注引用来源（如 [文档1]）
2. 如果上下文中没有相关信息，请如实说"根据现有资料，我无法回答这个问题"
3. 不要编造任何不在上下文中的信息
4. 用中文回答

【检索到的上下文】
{context}

【用户问题】
{question}

【回答】""")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


class RAGSystem:
    """RAG 系统封装类"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = get_retriever_for_rag(vectorstore)
        self.chain = create_rag_chain(self.retriever)

    def query(self, question: str) -> tuple[str, List[Document]]:
        """查询并返回（答案，来源文档）"""
        # 先检索
        docs = self.retriever.invoke(question)
        # 再生成
        answer = self.chain.invoke(question)
        return answer, docs


def get_retriever_for_rag(vectorstore, k: int = None):
    if k is None:
        k = RETRIEVAL_K
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
```

#### 5.3.5 Agent 工具定义 `tools.py`

```python
from langchain_core.tools import tool
from typing import Optional


# ===== 工具1：知识库检索 =====
# 这是一个全局变量，会在 agent.py 中注入 RAG 实例
_rag_system = None

def set_rag_system(rag_system):
    """设置全局 RAG 系统引用"""
    global _rag_system
    _rag_system = rag_system

@tool
def search_knowledge_base(query: str) -> str:
    """
    在本地知识库中搜索信息。当你需要查找文档中的具体事实、数据、
    或专业知识时使用此工具。输入应该是一个完整、明确的搜索问题。
    """
    if _rag_system is None:
        return "知识库尚未初始化，请先上传文档。"
    try:
        answer, docs = _rag_system.query(query)
        # 附上来源信息
        sources = set(doc.metadata.get("source", "未知") for doc in docs)
        source_str = "、".join(sources)
        return f"{answer}\n\n📚 参考来源：{source_str}"
    except Exception as e:
        return f"知识库搜索出错：{str(e)}"


# ===== 工具2：计算器 =====
@tool
def calculate(expression: str) -> str:
    """
    执行数学计算。当你需要进行精确的数学运算时使用此工具。
    输入为一个数学表达式，例如 '100 * 0.2 + 50'、'2**10'。
    """
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "int": int, "float": float
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# ===== 工具3：列出知识库中的文档 =====
@tool
def list_documents(dummy: str = "") -> str:
    """
    列出知识库中已有的所有文档。当用户询问"有哪些文档"、"知识库有什么"时使用。
    参数不需要填写，传空字符串即可。
    """
    if _rag_system is None:
        return "知识库尚未初始化。"
    try:
        # 从向量库中获取所有文档的 metadata
        results = _rag_system.vectorstore.get()
        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        if sources:
            return "知识库中包含以下文档：\n" + "\n".join(f"  - {s}" for s in sorted(sources))
        return "知识库为空。"
    except Exception as e:
        return f"获取文档列表出错：{str(e)}"


# ===== 工具4：文档数量统计 =====
@tool
def count_chunks(dummy: str = "") -> str:
    """
    统计知识库中的文档块数量。当用户询问知识库大小、文档数量时使用。
    """
    if _rag_system is None:
        return "知识库尚未初始化，文档块数量为 0。"
    try:
        count = _rag_system.vectorstore._collection.count()
        return f"知识库中共有 {count} 个文档块。"
    except Exception as e:
        return f"统计出错：{str(e)}"
```

#### 5.3.6 Agent 创建与管理 `agent.py`

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from tools import (
    search_knowledge_base,
    calculate,
    list_documents,
    count_chunks,
    set_rag_system
)


def create_qa_agent(rag_system):
    """创建知识库问答 Agent"""
    # 注入 RAG 系统到工具中
    set_rag_system(rag_system)

    # 初始化 LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.1,  # Agent 需要更低的温度以保证决策稳定
    )

    # 工具列表
    tools = [
        search_knowledge_base,
        calculate,
        list_documents,
        count_chunks,
    ]

    # System Prompt（定义 Agent 的行为规范）
    system_prompt = """你是一个智能知识库助手，名叫"小知"。

你可以使用以下工具来帮助用户：
- search_knowledge_base：在知识库中搜索信息（最重要的工具）
- calculate：进行数学计算
- list_documents：列出知识库中的文档
- count_chunks：查看知识库的规模

工作流程：
1. 当用户提问时，首先使用 search_knowledge_base 在知识库中搜索
2. 如果知识库没有相关信息，诚实地告诉用户
3. 如果问题涉及计算，使用 calculate 工具
4. 回答时引用知识库中的来源
5. 用中文回答，保持友好和专业

重要：你必须使用工具来获取信息，不要编造任何事实。"""

    # 创建记忆管理器
    memory = MemorySaver()

    # 创建 Agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt,
        checkpointer=memory,
    )

    return agent


class AgentManager:
    """Agent 管理器，封装 Agent 调用"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.agent = create_qa_agent(rag_system)
        self.sessions = {}  # 管理不同用户的会话

    def chat(self, message: str, session_id: str = "default") -> dict:
        """发送消息并获取回复"""
        config = {"configurable": {"thread_id": session_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )

        # 提取最终回答
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)

        # 提取中间步骤（Agent 的思考和工具调用）
        steps = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "type": "tool_call",
                        "tool": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    })

        return {
            "answer": answer,
            "steps": steps,  # Agent 的思考过程
        }

    def reset_session(self, session_id: str):
        """重置会话"""
        # 创建新的 agent 实例会重置该 session 的记忆
        pass
```

#### 5.3.7 Streamlit 前端 `main.py`

```python
import os
import streamlit as st
from document_loader import process_all_documents
from vector_store import create_vectorstore, load_vectorstore
from rag_chain import RAGSystem
from agent import AgentManager

# ===== 页面配置 =====
st.set_page_config(
    page_title="智能知识库问答系统",
    page_icon="📚",
    layout="wide",
)

st.title("📚 智能知识库问答系统")
st.caption("基于 LangChain + RAG + Agent 构建 | 上传文档 → 智能问答")

# ===== 侧边栏：文档管理 =====
with st.sidebar:
    st.header("📁 文档管理")

    # 上传文档
    uploaded_files = st.file_uploader(
        "上传 PDF 或 TXT 文档",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("处理并入库", type="primary"):
        with st.spinner("正在处理文档..."):
            # 保存上传的文件
            for f in uploaded_files:
                file_path = os.path.join("data/uploads", f.name)
                with open(file_path, "wb") as fp:
                    fp.write(f.getbuffer())

            # 处理文档
            documents = process_all_documents()
            if documents:
                # 创建向量数据库
                create_vectorstore(documents)
                st.success(f"已入库 {len(documents)} 个文本块")
                # 清除缓存，强制重新加载
                st.cache_resource.clear()
                st.rerun()

    st.divider()

    # 显示知识库状态
    st.header("📊 知识库状态")
    vectorstore = load_vectorstore()
    if vectorstore:
        count = vectorstore._collection.count()
        st.metric("文档块数量", count)

        # 显示文档列表
        results = vectorstore.get()
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        if sources:
            st.write("**已入库文档：**")
            for s in sorted(sources):
                st.text(f"  📄 {s}")
    else:
        st.info("知识库为空，请上传文档")

# ===== 主区域：聊天界面 =====

# 初始化系统
@st.cache_resource
def init_system():
    """初始化 RAG 系统和 Agent"""
    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.warning("知识库为空！请先在侧边栏上传文档。")
        st.stop()

    rag_system = RAGSystem(vectorstore)
    agent_manager = AgentManager(rag_system)
    return agent_manager

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "steps" in msg and msg["steps"]:
            with st.expander("查看 Agent 思考过程"):
                for step in msg["steps"]:
                    st.json(step)

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取回复
    with st.chat_message("assistant"):
        try:
            # 尝试初始化系统
            agent_manager = init_system()

            with st.spinner("思考中..."):
                result = agent_manager.chat(prompt)

            st.markdown(result["answer"])

            # 显示思考过程
            if result["steps"]:
                with st.expander("🔍 Agent 执行过程"):
                    for i, step in enumerate(result["steps"], 1):
                        st.write(f"**步骤 {i}：调用工具 `{step['tool']}`**")
                        st.code(step["args"], language="json")

            # 保存到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "steps": result["steps"],
            })

        except Exception as e:
            st.error(f"系统错误：{e}")
            st.info("请确保：\n1. 已上传文档并入库\n2. API Key 配置正确\n3. 网络连接正常")
```

#### 5.3.8 依赖文件 `requirements.txt`

```txt
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-chroma>=0.2.0
langgraph>=0.2.0
chromadb>=0.5.0
pypdf>=5.0.0
pymupdf>=1.24.0
sentence-transformers>=3.0.0
streamlit>=1.35.0
python-dotenv>=1.0.0
openai>=1.50.0
```

### 5.4 运行项目

```bash
# 1. 创建项目目录
mkdir knowledge_qa_system
cd knowledge_qa_system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 .env 文件（填入你的 API Key）
echo 'OPENAI_API_KEY=sk-xxxxx' > .env
echo 'OPENAI_BASE_URL=https://api.openai.com/v1' >> .env

# 4. 创建目录
mkdir -p data/uploads data/chroma_db

# 5. 放入你的 PDF 文档到 data/uploads/

# 6. 启动
streamlit run main.py

# 浏览器打开 http://localhost:8501 即可使用
```

---

## 6. 进阶技巧

### 6.1 使用国产大模型

```python
# Qwen 通义千问
from langchain_openai import ChatOpenAI

qwen_llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# DeepSeek
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# Zhipu 智谱 GLM
zhipu_llm = ChatOpenAI(
    model="glm-4",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4"
)
```

### 6.2 本地部署（完全免费）

```bash
# 安装 Ollama
# 从 https://ollama.com 下载安装

# 拉取模型
ollama pull llama3:8b              # LLM
ollama pull nomic-embed-text       # Embedding
```

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 本地 LLM
llm = ChatOllama(model="llama3:8b", temperature=0.1)

# 本地 Embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 其余代码和在线模型完全一样！
```

### 6.3 提高检索质量的 Checklist

1. **调整 chunk_size**：太小丢上下文，太大检索不准。1000 是好的起点
2. **设置 chunk_overlap**：让相邻块有重叠，避免关键信息被切断。通常设为 chunk_size 的 20%
3. **增加检索数量**：k=4 是起步，尝试 k=10 再配合重排序
4. **使用重排序**：粗检索 + Cross-Encoder 精排，显著提升精度
5. **优化 Prompt**：明确告诉 LLM "不知道就说不知道"，减少幻觉
6. **混合检索**：语义 + 关键词，覆盖更多情况
7. **评估迭代**：用 Ragas 等工具评估检索质量

---

## 7. 常见问题与排错

### Q1：提示 "知识库为空"
**解决**：先在侧边栏上传 PDF 文件，点击"处理并入库"，等待处理完成。

### Q2：回答一直在转圈
**可能原因**：
- API Key 未配置或已过期
- 网络连接问题（检查代理/VPN）
- API 额度用尽

### Q3：检索到的内容不相关
**解决**：
- 减小 chunk_size（如 500）
- 增加 chunk_overlap
- 使用重排序（Cross-Encoder Reranker）
- 尝试混合检索（语义 + BM25）

### Q4：Agent 调用工具后无响应
**解决**：
- 检查工具的 docstring 是否清晰明确
- 降低 LLM 的 temperature（0.1 左右）
- 增加 agent 的最大迭代次数

### Q5：ChromaDB 初始化慢
**解决**：
- ChromaDB 首次加载会下载依赖，需要耐心等待
- 如果一直卡住，尝试 `pip install chromadb --upgrade`

---

## 学习资源推荐

| 资源 | 说明 |
|------|------|
| [LangChain 官方文档](https://docs.langchain.com) | 最权威的参考 |
| [LangGraph 文档](https://langchain-ai.github.io/langgraph/) | Agent 编排框架 |
| [Microsoft LangChain 初学者课程](https://github.com/microsoft/langchain-for-beginners) | 9 章递进式教程 |
| [RAG from Scratch (GitHub)](https://github.com/pguso/rag-from-scratch) | 从零理解 RAG |
| [RAG Implementations (GitHub)](https://github.com/SamanCh/RAG_implementations) | 从基础到高级的 RAG 实现 |
| [JetBrains LangChain 2026 指南](https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/) | 2026 最新教程 |
| [华为云 RAG Agent 实践](https://bbs.huaweicloud.com/blogs/475312) | 中文实战教程 |

---

> **总结**：这套技术栈的核心思想是——**LangChain** 提供框架骨架，**RAG** 让 LLM 能访问外部知识，**Agent** 赋予系统自主推理和行动的能力。三者结合，可以构建出强大的企业级智能应用。

---

*文档生成时间：2026年5月27日*
