Feedback schema and helper module

Files added:
- `schemas/feedback_schema.json` - JSON Schema describing accepted feedback payload.
- `feedback.py` - Python dataclass + validation, feature extraction, scoring helpers.

Quick usage

1. Validate payload:

```py
from feedback import validate_feedback
ok, errors = validate_feedback(payload)
```

2. Extract features and compute score:

```py
from feedback import extract_features, compute_score
feats = extract_features(payload['text'], {'timestamp': payload['timestamp']})
score, score_raw, params = compute_score(payload['views'], payload['likes'], payload['comments'], decay_lambda=0.01, timestamp=payload['timestamp'])
```

3. Convert to DB row (for SQLite insertion):

```py
from feedback import Feedback, to_db_row
fb = Feedback(**payload)
fb.features = feats
fb.score = score
fb.score_raw = score_raw
fb.score_params = params
row = to_db_row(fb)
# insert row into feedback table
```

Notes

- `feedback.py` uses simple heuristics for features (hashtag/emoji/CTA counts). Replace sentiment/readability with external libs if needed.
- Adjust `DEFAULT_WEIGHTS` or pass custom weights into `compute_score`.
- Consider adding follower_count or impressions to payload if platform provides them.
