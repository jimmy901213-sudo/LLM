import sqlite3
import os
import json
from typing import Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feedback.sqlite3')


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def insert_feedback(row: Dict[str, Any]):
    conn = get_conn()
    cur = conn.cursor()
    def _ensure_meta(m):
        if m is None:
            return {}
        if isinstance(m, str):
            try:
                return json.loads(m)
            except Exception:
                return {}
        return m

    meta = _ensure_meta(row.get('metadata'))

    def _maybe_json(x):
        if x is None:
            return None
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x)
        except Exception:
            return str(x)

    cur.execute(
        '''INSERT OR REPLACE INTO feedback
        (id, text, views, likes, comments, timestamp, source, campaign, tags, author_raw, author_hash, author, language, score, score_raw, score_params, embedding_id, ab_group, features, quality_flags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            row.get('id'),
            row.get('text'),
            row.get('views', 0),
            row.get('likes', 0),
            row.get('comments', 0),
            row.get('timestamp'),
            row.get('source'),
            meta.get('campaign'),
            _maybe_json(meta.get('tags')),
            row.get('author_raw') or meta.get('author'),
            row.get('author_hash'),
            row.get('author_hash') or (row.get('author_raw') or meta.get('author')),
            meta.get('language'),
            row.get('score'),
            row.get('score_raw'),
            _maybe_json(row.get('score_params')),
            row.get('embedding_id'),
            row.get('ab_group'),
            _maybe_json(row.get('features')),
            _maybe_json(row.get('quality')),
        )
    )
    conn.commit()
    conn.close()
