Privacy, retention and guardrails

- Data minimization: only store identifiers required for analysis. Avoid storing PII such as emails or phone numbers.
- Anonymization: if author identifiers are provided, consider hashing them before saving in `author` field.
Anonymization: the system now stores `author_raw` (optional) and a deterministic `author_hash` by default. The ingestion pipeline
will save `author_raw` and `author_hash` into the feedback DB; `author_hash` is used for deterministic A/B bucketing.
If you prefer NOT to store `author_raw`, remove it from the payload or update the pipeline to discard it.
- Retention: define `retention_days` (e.g., 365). Implement periodic job to delete or anonymize old raw text.
- Consent: ensure you have consent to store and analyze user-generated content when applicable.
- Spam/bot filtering: implement heuristics or integrate with third-party services to mark suspicious records (e.g., extremely high likes/views in short time, repeated identical texts).
- Human-in-the-loop: automatic suggestions should be flagged as `draft` and require human approval prior to publishing.

Human tasks required:
- Decide retention policy (number of days)
- Decide whether to store original author identifiers or use anonymized hashes
- Approve privacy policy text for end-users
