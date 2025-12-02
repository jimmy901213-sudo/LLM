import json
import os
import shutil
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# 0. 定義持久化路徑
DB_PATH = "./chroma_db"

# 1. 初始化嵌入模型 (本地運行)
# 確保您已經運行了 `ollama pull nomic-embed-text`
print("正在連接到 Ollama 上的 nomic-embed-text 模型...")
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # 測試連接
    embeddings.embed_query("測試連接")
    print("Ollama 嵌入模型加載完畢。")
except Exception as e:
    print(f"錯誤：無法連接到 Ollama 或加載 'nomic-embed-text' 模型。")
    print(f"請確保 Ollama 正在運行並且您已經執行了 'ollama pull nomic-embed-text'。")
    print(f"詳細錯誤: {e}")
    exit()

# 2. 處理知識庫 A (產品事實)
print("正在加載 KB-A (products.json)...")
with open("./products.json", "r", encoding="utf-8") as f:
    product_entries = json.load(f)

docs_kb_a = []
for entry in product_entries:
    # 為了更好的檢索，我們將 JSON 結構扁平化為文本
    content = f"產品名稱: {entry['name']}\n"
    content += f"產品 ID: {entry['product_id']}\n"
    content += f"類別: {entry['category']}\n"
    content += f"功能: {', '.join(entry['features'])}\n"
    content += f"描述: {entry['description']}"
    
    # 創建一個新的 Document 對象
    new_doc = Document(page_content=content, metadata={
        "source": "product_db",  # 關鍵元數據
        "product_id": entry['product_id']
    })
    docs_kb_a.append(new_doc)
print(f"KB-A 加載完畢，共 {len(docs_kb_a)} 個產品。")

# 3. 處理知識庫 B (AIO/SEO 規則)
print("正在加載 KB-B (rules.md)...")
loader_kb_b = TextLoader("./rules.md", encoding="utf-8")
rule_docs_raw = loader_kb_b.load()

splitter_kb_b = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_kb_b = splitter_kb_b.split_documents(rule_docs_raw)

# 為規則文檔添加元數據
for doc in docs_kb_b:
    doc.metadata = {"source": "aio_rules"} # 關鍵元數據
print(f"KB-B 加載完畢，共 {len(docs_kb_b)} 條規則塊。")

# 4. 合併並存儲到 ChromaDB
all_docs = docs_kb_a + docs_kb_b

if os.path.exists(DB_PATH):
    print(f"正在刪除舊的 ChromaDB 目錄: {DB_PATH}")
    shutil.rmtree(DB_PATH)

print("正在創建新的 ChromaDB 向量數據庫 (這可能需要一點時間)...")
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print(f"向量數據庫創建完畢，共索引 {len(all_docs)} 個文檔塊。")
print(f"數據庫已保存至: {DB_PATH}")