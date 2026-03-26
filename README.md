# SecretOrNot — Privacy Router

Route what needs to be private. Prove it was.

SecretOrNot is a standalone privacy router that sits between any OpenAI-compatible LLM client and two destination endpoints. It classifies every incoming prompt as **PRIVATE** or **PUBLIC** and routes accordingly — transparently, with no client-side changes required.

- **PRIVATE** prompts → Secret AI (runs inside a TEE)
- **PUBLIC** prompts → any public LLM (OpenAI, etc.)

The router itself runs inside a Secret VM (TEE), so the classification decision is attested.

## How It Works

```
Any OpenAI-compatible client
        ↓
   POST /v1/chat/completions
        ↓
┌──────────────────────────┐
│   Layer 1: Rule filter   │  regex/keywords (~1ms)
│   Layer 2: Model class.  │  Gemma3-4B via Secret AI (~300ms)
│   Routing decision       │  fail-closed: default to PRIVATE
│   Audit log              │  metadata only, never prompts
└──────────────────────────┘
     ↓              ↓
  Secret AI     Public LLM
```

## Quick Start

```bash
# Clone and install
cd secretornot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your endpoints and API keys

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Point Any Client At It

The router exposes a standard OpenAI-compatible endpoint. Point your client's `base_url` at the router — no other changes needed.

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used-by-router",  # router uses its own keys
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
# → Classified PUBLIC, routed to OpenAI

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "My SSN is 123-45-6789, can you help me?"}],
)
# → Classified PRIVATE (rule: SSN pattern), routed to Secret AI
```

**curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

**Streaming:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a poem about the ocean"}]
  }'
```

## Classification Layers

### Layer 1: Rule-based filter (~1ms)
Pattern matching for known sensitive content. If any rule matches, the prompt is immediately classified as PRIVATE — Layer 2 is never called.

Rules cover: PII (SSN, credit cards, phone numbers, emails), medical terms, credentials/secrets, financial data, legal terms, and an explicit `[PRIVATE]` tag.

### Layer 2: Model classifier (~300ms)
For prompts that don't trigger any rule, a Gemma3-4B model (running on Secret AI) classifies the prompt as PRIVATE or PUBLIC with a confidence score.

If confidence is below the threshold (default 0.85), the prompt is classified as PRIVATE.

## Fail-Closed Policy

SecretOrNot **always fails toward PRIVATE**:
- Classifier timeout → PRIVATE
- Classifier error → PRIVATE
- Low confidence → PRIVATE
- Parse failure → PRIVATE

A prompt is only routed to the public LLM when the classifier is confident it contains no sensitive content.

## Audit Log

Every routing decision is logged to `audit.jsonl` (configurable via `AUDIT_LOG_FILE`). Logs contain **metadata only** — never prompt content.

```json
{
  "request_id": "...",
  "timestamp": "2025-03-25T12:00:00Z",
  "classification": "PRIVATE",
  "layer_triggered": "rule",
  "rule_matched": "ssn",
  "confidence": 1.0,
  "routed_to": "secret_ai",
  "model_requested": "gpt-4o",
  "latency_ms": {"classification": 0.5, "inference": 1200.0, "total": 1200.5}
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_PORT` | `8000` | Server port |
| `SECRET_AI_ENDPOINT` | required | Private LLM endpoint (Secret AI) |
| `SECRET_AI_API_KEY` | required | API key for Secret AI |
| `PUBLIC_LLM_ENDPOINT` | required | Public LLM endpoint (OpenAI, etc.) |
| `PUBLIC_LLM_API_KEY` | required | API key for public LLM |
| `CLASSIFIER_ENDPOINT` | `SECRET_AI_ENDPOINT` | Classifier model endpoint |
| `CLASSIFIER_API_KEY` | `SECRET_AI_API_KEY` | Classifier API key |
| `CLASSIFIER_MODEL` | `gemma3:4b` | Model name for Layer 2 classifier |
| `CLASSIFICATION_TIMEOUT_MS` | `2000` | Max time for classification before fail-to-private |
| `CONFIDENCE_THRESHOLD` | `0.85` | Below this → classify as PRIVATE |
| `DEFAULT_POLICY` | `private` | Fallback policy |
| `PROXY_TIMEOUT_S` | `120` | Timeout for LLM inference proxy |
| `AUDIT_LOG_FILE` | `audit.jsonl` | Audit log output file |
