import streamlit as st
import os
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from pydantic import BaseModel, Field
from typing import List, Dict

# --- 1. 定義 Pydantic Schema (我們的結構化輸出) ---
# 這個 Schema 將強制 Llama 3 以我們想要的 JSON 格式回應 [3, 4]
class MarketingContent(BaseModel):
    product_name: str = Field(description="產品的官方全名")
    catchy_title: str = Field(description="優化的 AIO/SEO 標題，不超過 60 個字符")
    experience_paragraph: str = Field(description="E-E-A-T 化的第一人稱使用經驗段落，需結合一個真實場景")
    features_bullets: List[str] = Field(description="從產品事實中提取的 3-5 個核心功能列表")
    semantic_tags: List[str] = Field(description="相關的語義關鍵字和實體 (例如 '戶外', '派對')")
    qa_pairs: List[Dict[str, str]] = Field(description="2-3 個 Q&A 對，格式為 [{'q': '...', 'a': '...'}]")

# --- 2. 系統初始化 (緩存以提高效能) ---
@st.cache_resource
def load_system():
    # 檢查 ChromaDB 是否存在
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        return None, None, None, "ChromaDB 目錄未找到。請先運行 'create_vectorstore.py'。"

    try:
        # 1. 初始化 LLM (結構化輸出)
        # 我們將 Pydantic 模型綁定到 Llama 3 [3, 2]
        llm = ChatOllama(model="llama3:8b", temperature=0.1)
        llm_structured = llm.with_structured_output(MarketingContent)
        
        # 2. 初始化嵌入模型
        # 確保 `ollama pull nomic-embed-text` 已經運行
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 3. 加載向量數據庫 [5, 6]
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # 4. 創建兩個檢索器 (實現雙重知識庫架構)
        retriever_products = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"source": "product_db"}, "k": 1} # 僅檢索KB-A，只返回最匹配的 1 個產品
        )
        retriever_rules = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"source": "aio_rules"}, "k": 5} # 檢索KB-B，返回 5 條相關規則
        )

        return llm_structured, retriever_products, retriever_rules, None
    except Exception as e:
        return None, None, None, f"系統初始化失敗：{e}。請確保 Ollama 正在運行，並且您已拉取 'llama3:8b' 和 'nomic-embed-text' 模型。"

# --- 3. Streamlit 界面 [7] ---
st.set_page_config(layout="wide")
st.title("🚀 AIO/SEO 行銷內容生成器 (本地版)")
st.caption("一個使用 Llama 3、RAG 和 Pydantic 的零預算大學生專案")

# 加載系統
structured_llm, prod_retriever, rule_retriever, error_msg = load_system()

if error_msg:
    st.error(error_msg)
else:
    st.sidebar.header("設置")
    product_query = st.sidebar.text_input("輸入產品名稱 (例如: X-100 音箱 或 Z-500 耳機)")
    
    if st.sidebar.button("生成 AIO/SEO 優化內容"):
        if not product_query:
            st.sidebar.error("請輸入產品名稱")
        else:
            with st.spinner("系統正在思考... (本地 Llama 3 8B 運行中，請耐心等待...)"):
                
                # --- RAG 核心邏輯 ---
                
                # 1. 檢索 KB-A (產品)
                product_context_docs = prod_retriever.invoke(product_query)
                product_context = "\n---\n".join([doc.page_content for doc in product_context_docs])
                
                # 2. 檢索 KB-B (規則) - 查詢是固定的，我們需要所有相關規則
                rule_context_docs = rule_retriever.invoke("所有 AIO/SEO/E-E-A-T 行銷規則")
                rule_context = "\n---\n".join([doc.page_content for doc in rule_context_docs])
                
                # 3. 創建 RAG 提示詞 [8, 9]
                # (從 Pydantic Schema 中獲取 JSON Schema 描述以指導 LLM [3])
                json_schema_description = MarketingContent.model_json_schema()
                
                prompt = f"""
                你是一名專業的電商行銷內容撰寫專家，精通 AIO (AI 優化) 和 Google E-E-A-T 規則。

                **你的任務**：
                根據下方提供的「產品事實」和「行銷規則」，為該產品生成優化的行銷內容。
                你必須嚴格按照「輸出格式」要求，僅返回一個 JSON 對象，不要包含任何解釋或額外的文本。

                ---
                [上下文 1：產品事實]
                {product_context}
                ---
                [上下文 2：行銷規則]
                {rule_context}
                ---
                [用戶查詢]: "為 {product_query} 生成完整的 AIO 優化行銷內容"
                ---
                [輸出格式]: 請嚴格遵循此 JSON Schema: {json_schema_description}
                ---
                """
                
                # --- Streamlit 作為調試器 ---
                with st.expander("🔍 點此查看 RAG 系統的『思考過程』"):
                    st.subheader("檢索到的 [產品事實]:")
                    st.text(product_context)
                    st.subheader("檢索到的 [行銷規則]:")
                    st.text(rule_context)
                    st.subheader("完整的 RAG 提示詞 (發送給 Llama 3):")
                    st.text(prompt)
                
                # 4. 調用結構化 LLM 
                try:
                    #.invoke() 會返回一個 Pydantic 對象，而不是原始字符串
                    response_obj = structured_llm.invoke(prompt) 
                    
                    st.subheader(f"✅ 為 {response_obj.product_name} 生成的內容：")
                    
                    st.markdown(f"### {response_obj.catchy_title}")
                    st.divider()
                    
                    st.markdown("#### E-E-A-T 經驗段落:")
                    st.markdown(response_obj.experience_paragraph)
                    
                    st.markdown("#### 核心功能 (AIO 列表):")
                    st.markdown("\n".join(f"- {item}" for item in response_obj.features_bullets))
                    
                    st.markdown("#### Q&A 部分:")
                    for pair in response_obj.qa_pairs:
                        st.markdown(f"**Q: {pair['q']}**")
                        st.markdown(f"A: {pair['a']}")
                    
                    st.markdown("#### 語義標籤:")
                    st.markdown(", ".join(response_obj.semantic_tags))
                        
                    st.markdown("---")
                    st.subheader("原始 JSON 輸出 (用於 API):")
                    st.json(response_obj.model_dump_json())
                    
                except Exception as e:
                    st.error(f"本地 LLM 輸出錯誤：{e}")
                    st.error("Llama 3 8B 未能正確生成 Pydantic 結構。請嘗試重啟 Ollama 或檢查提示詞。")