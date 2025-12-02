from flask import Flask, request, jsonify
import hashlib
from feedback import validate_feedback, extract_features, compute_score, Feedback, to_db_row, hash_author
from db.db import insert_feedback
import traceback
import os

# Chroma / Ollama imports
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')

app = Flask(__name__)

# initialize embeddings and vectorstore lazily
_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        emb = get_embeddings()
        _vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=emb)
    return _vectorstore


def assign_ab_group(author_hash: str, seed: int = 100) -> str:
    """
    Deterministic bucketing by author_hash to avoid the same author being in multiple groups.
    Returns one of: 'control', 'treatment_a', 'treatment_b'
    Buckets: control 50%, treatment_a 30%, treatment_b 20% (configurable by changing thresholds).
    """
    if not author_hash:
        # fallback random-like assignment using current time hash
        import time
        hval = int(hashlib.sha256(str(time.time()).encode('utf-8')).hexdigest()[:8], 16)
    else:
        hval = int(author_hash[:8], 16)
    pct = hval % 100
    if pct < 50:
        return 'control'
    if pct < 80:
        return 'treatment_a'
    return 'treatment_b'


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/feedback', methods=['POST'])
def ingest_feedback():
    payload = request.get_json()
    if payload is None:
        return jsonify({'error': 'invalid-json'}), 400

    items = payload if isinstance(payload, list) else [payload]
    results = []
    for item in items:
        try:
            ok, errors = validate_feedback(item)
            if not ok:
                results.append({'id': item.get('id'), 'status': 'invalid', 'errors': errors})
                continue

            # extract features and compute score
            feats = extract_features(item['text'], {'timestamp': item.get('timestamp')})
            score, score_raw, params = compute_score(
                item.get('views', 0), item.get('likes', 0), item.get('comments', 0),
                weights=None, normalization='log1p', decay_lambda=0.01, timestamp=item.get('timestamp')
            )

            # handle author id: store raw and hashed value; deterministic A/B assignment by author
            author_raw = None
            if isinstance(item.get('metadata'), dict):
                author_raw = item.get('metadata', {}).get('author')
            if not author_raw:
                author_raw = item.get('author') or item.get('author_raw')

            author_hash = hash_author(author_raw)
            ab_group = assign_ab_group(author_hash)

            fb = Feedback(**item)
            fb.author_raw = author_raw
            fb.author_hash = author_hash
            fb.ab_group = ab_group
            fb.features = feats
            fb.score = score
            fb.score_raw = score_raw
            fb.score_params = params

            # insert into sqlite
            row = to_db_row(fb)
            insert_feedback(row)

            # add to chroma vectorstore
            try:
                vectorstore = get_vectorstore()
                doc = Document(page_content=item['text'], metadata={
                    'raw_id': item['id'],
                    'score': score,
                    'timestamp': item.get('timestamp'),
                    'source': item.get('source'),
                    'author_hash': author_hash,
                    'ab_group': ab_group,
                })
                vectorstore.add_documents([doc])
                vectorstore.persist()
                # Note: Chroma may generate an internal id; we can't easily get it here across implementations
                fb.embedding_id = item['id']
                # update DB with embedding_id
                insert_feedback(to_db_row(fb))
            except Exception as e:
                # log embedding error but don't fail the whole request
                print('Embedding error:', e)

            results.append({'id': item.get('id'), 'status': 'inserted', 'score': score})

        except Exception as e:
            traceback.print_exc()
            results.append({'id': item.get('id') if isinstance(item, dict) else None, 'status': 'error', 'error': str(e)})

    return jsonify({'results': results})


if __name__ == '__main__':
    # create DB if not exists
    try:
        from db.init_db import ensure_db
        ensure_db()
    except Exception as e:
        print('Warning: failed to init DB', e)

    app.run(host='0.0.0.0', port=8080)
