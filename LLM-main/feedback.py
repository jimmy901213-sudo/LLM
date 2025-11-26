from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List
import datetime
import math
import json
import re
import hashlib


@dataclass
class Feedback:
    id: str
    text: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    source: str = "threads"
    metadata: Dict[str, Any] = field(default_factory=dict)
    score_raw: Optional[float] = None
    score: Optional[float] = None
    score_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    embedding_id: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    author_hash: Optional[str] = None
    author_raw: Optional[str] = None
    ab_group: Optional[str] = None


# Validation
def validate_feedback(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(payload, dict):
        return False, ["payload-must-be-object"]
    if "id" not in payload or not isinstance(payload.get("id"), str) or not payload.get("id"):
        errors.append("missing-or-invalid-id")
    if "text" not in payload or not isinstance(payload.get("text"), str) or len(payload.get("text").strip()) < 3:
        errors.append("missing-or-invalid-text")

    for k in ("views", "likes", "comments"):
        v = payload.get(k, 0)
        if not isinstance(v, int) or v < 0:
            errors.append(f"{k}-must-be-nonnegative-int")

    ts = payload.get("timestamp")
    if ts:
        try:
            datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            errors.append("invalid-timestamp-format")
    else:
        errors.append("missing-timestamp")

    if "source" not in payload or not isinstance(payload.get("source"), str):
        errors.append("missing-or-invalid-source")

    return (len(errors) == 0), errors


# Feature extraction
def extract_features(text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not text:
        return {}
    features: Dict[str, Any] = {}
    features["char_count"] = len(text)
    features["word_count"] = len(re.findall(r"\w+", text, flags=re.UNICODE))
    features["hashtag_count"] = len(re.findall(r"#\w+", text))

    # simple emoji regex covering common ranges
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",
        flags=re.UNICODE,
    )
    features["emoji_count"] = len(emoji_pattern.findall(text))

    features["url_present"] = bool(re.search(r"https?://", text))

    # CTA count heuristic
    cta_words = ["buy", "click", "now", "limited", "立即", "馬上", "現在", "搶購", "立刻", "點擊"]
    text_lower = text.lower()
    features["cta_count"] = sum(1 for w in cta_words if w in text_lower)

    features["newline_count"] = text.count("\n")
    features["caps_ratio"] = sum(1 for c in text if c.isupper()) / max(1, features["char_count"])

    # time context if provided
    if metadata:
        try:
            if "timestamp" in metadata:
                ts = metadata.get("timestamp")
                dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                features["hour_of_day"] = dt.hour
                features["weekday"] = dt.weekday()
        except Exception:
            pass

    # placeholders for sentiment/readability (can be filled with external libs)
    features["sentiment"] = None
    features["readability"] = None

    return features


# Scoring
DEFAULT_WEIGHTS = {"views": 0.5, "likes": 0.3, "comments": 0.2}


def _normalize(value: int, method: str = "log1p") -> float:
    if method == "log1p":
        return math.log1p(value)
    if method == "identity":
        return float(value)
    # fallback
    return math.log1p(value)


def compute_score(
    views: int,
    likes: int,
    comments: int,
    weights: Optional[Dict[str, float]] = None,
    normalization: str = "log1p",
    decay_lambda: float = 0.0,
    timestamp: Optional[str] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (score, score_raw, params)
    - score_raw: weighted normalized sum (no decay)
    - score: score_raw * decay_factor (if decay_lambda > 0 and timestamp provided)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    v_n = _normalize(views, normalization)
    l_n = _normalize(likes, normalization)
    c_n = _normalize(comments, normalization)

    score_raw = weights.get("views", 0) * v_n + weights.get("likes", 0) * l_n + weights.get("comments", 0) * c_n

    decay_factor = 1.0
    if decay_lambda and timestamp:
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            age_days = (datetime.datetime.utcnow() - dt).total_seconds() / 86400.0
            decay_factor = math.exp(-decay_lambda * age_days)
        except Exception:
            decay_factor = 1.0

    score = score_raw * decay_factor
    params = {"weights": weights, "normalization": normalization, "decay_lambda": decay_lambda}
    return score, score_raw, params


# Helpers to convert to DB-friendly representation
def to_db_row(fb: Feedback) -> Dict[str, Any]:
    row = asdict(fb)
    # ensure JSON serializable fields
    row["features"] = json.dumps(row.get("features") or {})
    row["metadata"] = json.dumps(row.get("metadata") or {})
    row["score_params"] = json.dumps(row.get("score_params") or {})
    # include explicit author fields (raw + hash) if present
    if fb.author_raw:
        row["author_raw"] = fb.author_raw
    if fb.author_hash:
        row["author_hash"] = fb.author_hash
    if fb.ab_group:
        row["ab_group"] = fb.ab_group
    return row


def hash_author(author_id: Optional[str]) -> Optional[str]:
    if not author_id:
        return None
    h = hashlib.sha256(author_id.encode('utf-8')).hexdigest()
    return h


if __name__ == "__main__":
    # quick local test
    sample = {
        "id": "ad_20251119_01",
        "text": "超值優惠！限時 48 小時，點擊搶購 ➜ #優惠 #限時 http://example.com",
        "views": 12000,
        "likes": 430,
        "comments": 52,
        "timestamp": "2025-11-18T12:00:00Z",
        "source": "threads",
        "metadata": {"campaign": "holiday_sale"},
    }
    ok, errs = validate_feedback(sample)
    print("valid:", ok, "errors:", errs)
    feats = extract_features(sample["text"], {"timestamp": sample["timestamp"]})
    s, s_raw, params = compute_score(sample["views"], sample["likes"], sample["comments"], decay_lambda=0.01, timestamp=sample["timestamp"])
    fb = Feedback(**sample)
    fb.features = feats
    fb.score = s
    fb.score_raw = s_raw
    fb.score_params = params
    print("score:", s)
    print("db_row sample keys:", list(to_db_row(fb).keys()))
