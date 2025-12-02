Run instructions (PowerShell)

1) Initialize DB:

```powershell
python db\init_db.py
```

2) Start ingestion API (Flask):

```powershell
# (optional) create virtualenv and install requirements
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run API
python api_feedback.py
```

3) Sample ingestion (PowerShell curl):

```powershell
$payload = @{
  id = 'ad_20251119_01'
  text = '超值優惠！限時 48 小時，點擊搶購 ➜ #優惠 #限時 http://example.com'
  views = 12000
  likes = 430
  comments = 52
  timestamp = '2025-11-18T12:00:00Z'
  source = 'threads'
  metadata = @{ campaign = 'holiday_sale'; tags = @('促銷','短期'); author = 'creator_123' }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://localhost:8080/feedback -Method Post -Body $payload -ContentType 'application/json'
```

4) Suggest improvements (if Ollama is available):

```powershell
python suggest_improvements.py --id ad_20251119_01
```

Notes
- Ollama must be running and models `nomic-embed-text` and `llama3:8b` pulled for embeddings and LLM calls.
- If Ollama is not available, the suggest_improvements script will print a prepared prompt you can paste into your LLM UI.
