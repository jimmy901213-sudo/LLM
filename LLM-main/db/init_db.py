import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feedback.sqlite3')

SCHEMA = '''
CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  views INTEGER NOT NULL DEFAULT 0,
  likes INTEGER NOT NULL DEFAULT 0,
  comments INTEGER NOT NULL DEFAULT 0,
  timestamp TEXT NOT NULL,
  source TEXT,
  campaign TEXT,
  tags TEXT,
  author_raw TEXT,
  author_hash TEXT,
  author TEXT,
  language TEXT,
  score REAL,
  score_raw REAL,
  score_params TEXT,
  embedding_id TEXT,
  ab_group TEXT,
  features TEXT,
  quality_flags TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_feedback_score ON feedback(score DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
'''


def ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    ensure_db()
    print('Initialized DB at', DB_PATH)
