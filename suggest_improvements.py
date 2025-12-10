"""
Suggest improved ad variants using retrieval of positive/negative examples and an LLM.
If `ChatOllama` is available and Ollama is running, the script will call the model; otherwise it will print the prepared prompt for manual use.

Usage:
  python suggest_improvements.py --id ad_20251119_01

This script expects the Chroma DB at ./chroma_db and the feedback SQLite DB at ./data/feedback.sqlite3.
"""
import argparse
import json
import os
from typing import List

try:
    from langchain_ollama.chat_models import ChatOllama
    from langchain_community.vectorstores import Chroma
    from langchain_ollama.embeddings import OllamaEmbeddings
    from langchain_core.documents import Document
    LLAMA_AVAILABLE = True
except Exception:
    LLAMA_AVAILABLE = False


DB_PATH = './chroma_db'


def retrieve_examples(target_id: str, k_pos=3, k_neg=3):
    # if Chroma is available, perform similarity search; else return empty lists
    if not LLAMA_AVAILABLE:
        print('Ollama/Chroma not available; cannot retrieve examples automatically.')
        return [], []

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # retrieve similar documents to target (k_pos + k_neg)
    try:
        # find the document by metadata raw_id
        # Chroma retriever filter requires provider-specific usage; here we perform a general similarity search
        docs = vectorstore.similarity_search_by_vector(embeddings.embed_query(target_id), k=k_pos + k_neg)
    except Exception:
        # fall back to retrieving all and filtering
        docs = vectorstore.get()['documents'] if hasattr(vectorstore, 'get') else []

    # naive split by score metadata
    pos, neg = [], []
    for d in docs:
        m = getattr(d, 'metadata', {})
        score = m.get('score', 0)
        if score and score > 0:
            pos.append(d)
        else:
            neg.append(d)
    return pos[:k_pos], neg[:k_neg]


def build_prompt(target_text: str, positives: List[Document], negatives: List[Document]):
    prompt = '你是一位社群內容優化專家。\n'
    prompt += '目標內容：\n' + target_text + '\n\n'
    if positives:
        prompt += '高效範例（請分析為何受歡迎）:\n'
        for i, p in enumerate(positives, 1):
            prompt += f'範例 {i}: {p.page_content}\n元數據: {json.dumps(getattr(p, "metadata", {}), ensure_ascii=False)}\n\n'
    if negatives:
        prompt += '低效範例（請分析缺點）:\n'
        for i, n in enumerate(negatives, 1):
            prompt += f'範例 {i}: {n.page_content}\n元數據: {json.dumps(getattr(n, "metadata", {}), ensure_ascii=False)}\n\n'
    prompt += (
        '請基於上述資料給出：\n'
        '1) 針對目標內容的 3 個改進建議（每項需說明原因與預期提升項目，例如增加點擊率或互動），\n'
        '2) 基於高效範例抽取出 5 個具體可執行的寫作規則（例如使用短 CTA、加 Emoji 等），\n'
        '3) 產出 3 個可直接複製貼上的改寫版本 (Chinese Traditional)。\n'
    )
    return prompt


def call_llm(prompt: str):
    if not LLAMA_AVAILABLE:
        print('LLM not available. Here is the prompt you can use manually:')
        print('---')
        print(prompt)
        print('---')
        return

    llm = ChatOllama(model='llama3:8b', temperature=0.2)
    resp = llm.invoke(prompt)
    print('LLM Response:')
    print(resp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, help='target feedback id')
    args = parser.parse_args()

    # naive: load text by id from feedback DB
    import sqlite3
    conn = sqlite3.connect('./data/feedback.sqlite3')
    cur = conn.cursor()
    cur.execute('SELECT text FROM feedback WHERE id = ?', (args.id,))
    row = cur.fetchone()
    if not row:
        print('Target id not found in feedback DB')
        return
    target_text = row[0]

    pos, neg = retrieve_examples(args.id)
    prompt = build_prompt(target_text, pos, neg)
    call_llm(prompt)


if __name__ == '__main__':
    main()
