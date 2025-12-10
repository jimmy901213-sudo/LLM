import os
from typing import Tuple, Optional
from pydantic import BaseModel

try:
    from langchain_ollama.chat_models import ChatOllama
    from langchain_ollama.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
except Exception:
    # Imports may fail in environments without these packages — callers should handle errors
    ChatOllama = None  # type: ignore
    OllamaEmbeddings = None  # type: ignore
    Chroma = None  # type: ignore


def init_llm_and_retrievers(db_path: str = "./chroma_db") -> Tuple[Optional[object], Optional[object], Optional[object], Optional[str]]:
    """
    初始化本地 LLM 與兩個檢索器（產品檢索、規則檢索）。

    Returns:
        (structured_llm, retriever_products, retriever_rules, error_msg)
    """
    if not os.path.exists(db_path):
        return None, None, None, "ChromaDB 目錄未找到。"

    if ChatOllama is None or OllamaEmbeddings is None or Chroma is None:
        return None, None, None, "必要套件缺失：無法導入 Ollama 或 Chroma 相關套件。"

    try:
        # Initialize LLM and structured output wrapper will be done by caller (they have the Pydantic schema)
        llm = ChatOllama(model="llama3:8b", temperature=0.1)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # load Chroma vectorstore
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # create two retrievers
        retriever_products = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"source": "product_db"}, "k": 1}
        )
        retriever_rules = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"filter": {"source": "aio_rules"}, "k": 5}
        )

        return llm, retriever_products, retriever_rules, None
    except Exception as e:
        return None, None, None, f"系統初始化失敗：{e}"
