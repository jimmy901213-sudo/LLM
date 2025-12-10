"""
搜索模塊：為主程式提供產品搜索與檢索器上下文彙整的輔助函數
"""
from typing import List


def search_products(vectorstore_updater, keyword: str, limit: int = 10) -> List[dict]:
    """Thin wrapper for VectorstoreUpdater.search_products"""
    if vectorstore_updater is None:
        return []
    try:
        return vectorstore_updater.search_products(keyword, limit=limit)
    except Exception:
        return []


def get_documents_from_retriever(retriever, query: str, k: int = 1):
    """
    尝试通过 retriever.invoke 或 retriever.get_relevant_documents 获取文档列表。
    返回一个文档对象列表（可能是 langchain Document），或空列表。
    """
    if retriever is None:
        return []
    try:
        # prefer invoke (used in existing code)
        docs = retriever.invoke(query)
        return docs or []
    except Exception:
        try:
            # fallback for other retriever implementations
            docs = retriever.get_relevant_documents(query)
            return docs or []
        except Exception:
            return []


def combine_documents_content(docs: List[object]) -> str:
    """Combine page_content from a list of document-like objects."""
    if not docs:
        return ""
    parts = []
    for d in docs:
        try:
            text = getattr(d, "page_content", None)
            if text is None and isinstance(d, dict):
                text = d.get("page_content") or d.get("content")
            if text:
                parts.append(text)
        except Exception:
            continue
    return "\n---\n".join(parts)
