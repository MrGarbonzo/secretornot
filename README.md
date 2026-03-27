# SecretOrNot — Privacy Router

Route what needs to be private. Prove it was.

SecretOrNot is a privacy router that sits between any OpenAI-compatible LLM client and two destination endpoints. It classifies every incoming prompt as **PRIVATE** or **PUBLIC** and routes accordingly — no client-side changes required.

- **PRIVATE** prompts go to Secret AI (confidential compute / TEE)
- **PUBLIC** prompts go to any public LLM (Groq, OpenAI, etc.)

The router runs inside a Secret VM (Intel TDX), so the classification decision is attested.

## How It Works

```
Any OpenAI-compatible client
        |
   POST /v1/chat/completions
        |
+----------------------------+
|  Layer 1: Rule filter      |  regex/keywords (~1ms)
|  Layer 2: DistilBERT       |  fine-tuned ONNX classifier (~35ms)
|  Conversation-aware        |  classifies each message individually
|  Routing decision          |  fail-closed: default to PRIVATE
|  Audit log                 |  metadata only, never prompts
+----------------------------+
     |              |
  Secret AI     Public LLM
```

## Quick Start

```bash
# Clone and configure
cd secretornot
cp .env.example .env
# Edit .env with your endpoints and API keys

# Run with Docker
docker compose up -d

# Or run directly
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
    model="auto",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
# -> Classified PUBLIC, routed to public LLM

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "My SSN is 123-45-6789, can you help me?"}],
)
# -> Classified PRIVATE (rule: SSN pattern), routed to Secret AI
```

**curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

## Test UI

The router serves a test UI at `/` with two modes:

- **Chat** — full conversation with classification + LLM routing. A "New Chat" button resets conversation state.
- **Classifier** — test DistilBERT directly on single inputs. Each result has Correct/Wrong buttons that save labeled feedback for retraining.

## Classification

### Layer 1: Rules (~1ms)
Pattern matching for known sensitive content. If any rule matches, the prompt is immediately classified as PRIVATE — Layer 2 is never called.

Covers: PII (SSN, credit cards, phone numbers, emails), medical terms, credentials/secrets, financial data, legal terms, personal identifiers, and an explicit `[PRIVATE]` tag.

### Layer 2: DistilBERT (~35ms)
A fine-tuned DistilBERT model (66M parameters) runs locally via ONNX on CPU. It classifies prompts that don't trigger any rule. If confidence is below the threshold (default 0.85), the prompt is classified as PRIVATE.

### Conversation-Aware Routing

The `/v1/chat/completions` endpoint classifies each message in the conversation individually rather than truncating a concatenated blob at 256 tokens:

- Layer 1 (rules) scans every message — if any match, the entire conversation is PRIVATE
- Layer 2 (DistilBERT) runs on each user message newest-first, short-circuiting on the first PRIVATE result
- Once a conversation becomes private, it stays private until the client starts a new conversation

## Fail-Closed

SecretOrNot always fails toward PRIVATE:
- Classifier error -> PRIVATE
- Low confidence -> PRIVATE
- Parse failure -> PRIVATE
- No user messages -> PRIVATE

A prompt is only routed to the public LLM when the classifier is confident it contains no sensitive content.

## Feedback and Retraining

The Classifier tab in the test UI saves correct/incorrect labels to `feedback.jsonl`. To retrain:

```bash
# 1. Extract feedback from the deployed instance
scp -i secretvm_key root@<host>:/app/feedback.jsonl ./feedback.jsonl

# 2. Retrain (merges training_data.jsonl + feedback.jsonl)
python training/train.py

# 3. Export updated ONNX model
python training/export_onnx.py

# 4. Rebuild and deploy
docker build -t secretornot:local .
```

## Audit Log

Every routing decision is logged to `audit.jsonl`. Logs contain metadata only — never prompt content.

```json
{
  "request_id": "...",
  "timestamp": "2026-03-27T02:11:46Z",
  "classification": "PRIVATE",
  "layer_triggered": "rule",
  "rule_matched": "ssn",
  "confidence": 1.0,
  "routed_to": "secret_ai",
  "model_requested": "auto",
  "latency_ms": {"classification": 0.5, "inference": 1200.0, "total": 1200.5}
}
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible routing endpoint (conversation-aware) |
| `/classify` | POST | Classification only — accepts `{text}` or `{messages}` |
| `/feedback` | POST | Save correct/incorrect label for retraining |
| `/health` | GET | Health check |
| `/` | GET | Test UI (Chat + Classifier tabs) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_AI_ENDPOINT` | required | Private LLM endpoint (Secret AI) |
| `SECRET_AI_API_KEY` | required | API key for Secret AI |
| `PUBLIC_LLM_ENDPOINT` | required | Public LLM endpoint |
| `PUBLIC_LLM_API_KEY` | required | API key for public LLM |
| `DISTILBERT_MODEL_PATH` | `training/model.onnx` | Path to ONNX classifier model |
| `CONFIDENCE_THRESHOLD` | `0.85` | Below this -> classify as PRIVATE |
| `PROXY_TIMEOUT_S` | `120` | Timeout for LLM inference proxy |
| `AUDIT_LOG_FILE` | `audit.jsonl` | Audit log output file |
| `SECRET_AI_DEFAULT_MODEL` | `qwen3:8b` | Fallback model for Secret AI |
| `PUBLIC_LLM_DEFAULT_MODEL` | `llama-3.3-70b-versatile` | Fallback model for public LLM |

## Deployment

SecretOrNot is designed to run inside a Secret VM (Intel TDX) where:
- All classification happens within the TEE — decisions are attested
- Private prompts never leave the secure environment
- The DistilBERT classifier runs in-process with no network dependency
- Audit logs record metadata only, never prompt content
