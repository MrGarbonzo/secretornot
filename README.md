# SecretOrNot

**Route what needs to be private. Prove it was.**

SecretOrNot is a standalone privacy router that sits between any OpenAI-compatible
LLM client and two destination endpoints. It automatically classifies every prompt
as PRIVATE or PUBLIC and routes accordingly — with no changes required on the
client side.

- **PRIVATE** prompts route to [Secret AI](https://scrt.network) — confidential
  compute inside a TEE (Intel TDX). The prompt never leaves the secure enclave.
- **PUBLIC** prompts route to any public LLM (OpenAI, Groq, etc.) — fast and cheap.

The router itself runs inside Secret VM, so the routing decision is TEE-attested.
Your developers don't have to think about what's private. The router decides.
And you can prove it.

---

## How It Works

```
Any OpenAI-compatible client
           |
  POST /v1/chat/completions
           |
+---------------------------------------------+
|         SecretOrNot Router                   |  <- Runs inside Secret VM (TEE)
|                                             |
|  Layer 1: Rule filter        (~1ms)         |  regex + keyword matching
|  Layer 2: DistilBERT ONNX   (~35ms)        |  fine-tuned binary classifier
|  Routing decision                           |  fail-closed: default PRIVATE
|  Audit log                                  |  metadata only, never prompts
+---------------------------------------------+
           |                    |
       Secret AI            Public LLM
    (TEE-attested)       (OpenAI, Groq, etc.)
```

Every request is classified before any data leaves the router. Private prompts
stay inside the TEE. Public prompts are routed to the fastest available endpoint.
The client receives a standard OpenAI-format response either way.

---

## Quick Start

```bash
git clone https://github.com/your-org/secretornot
cd secretornot

cp .env.example .env
# Edit .env — add your Secret AI and public LLM credentials

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up -d
```

Verify it's running:

```bash
curl http://localhost:8000/health
```

---

## Integration — One Line Change

Point your existing LLM client at the router. Nothing else changes.

**Python:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://son.attestai.io:8000/v1",
    api_key="not-used",  # router manages its own keys
)

# General question — routed to public LLM automatically
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain how TCP works"}],
)

# Sensitive prompt — routed to Secret AI automatically
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "My SSN is 123-45-6789, help me file taxes"}],
)
```

**curl:**
```bash
curl http://son.attestai.io:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

The response is a standard OpenAI-format object. The client has no knowledge
of which endpoint was used.

---

## Classification

### Layer 1 — Rule Filter (~1ms)

Pattern matching against known sensitive signals. If any rule matches, the
prompt is immediately classified as PRIVATE. Layer 2 is never called.

| Category | Examples |
|---|---|
| PII | SSN, credit card numbers, phone numbers |
| Medical | diagnosis, patient, prescription, symptoms |
| Credentials | password, api_key, token, private_key |
| Financial | account number, routing number, salary |
| Legal | confidential, attorney-client, privileged |
| Identifiers | passport, driver's license, date of birth |
| Explicit flag | `[PRIVATE]` — inject this tag to force private routing |

### Layer 2 — DistilBERT ONNX (~35ms)

A fine-tuned DistilBERT model (66M parameters) runs locally in-process via
ONNX Runtime on CPU. It classifies prompts that don't trigger any Layer 1 rule.

The model is trained to catch what rules miss:
- Implicit business sensitivity ("our Q3 revenue was...")
- Personal context without keywords ("what should I do about my situation")
- Mixed prompts that blend public questions with private context
- HR and personnel topics without explicit markers

If confidence falls below `CONFIDENCE_THRESHOLD` (default 0.85), the prompt
is classified as PRIVATE.

### Conversation-Aware Routing

The router classifies each message in the conversation individually:

- Layer 1 scans every message — one match makes the whole conversation PRIVATE
- Layer 2 runs on user messages newest-first, short-circuits on first PRIVATE
- Once private, a conversation stays private until a new conversation starts

### Fail-Closed Policy

SecretOrNot always fails toward PRIVATE:

| Condition | Result |
|---|---|
| Classifier error | PRIVATE |
| Timeout | PRIVATE |
| Low confidence | PRIVATE |
| Parse failure | PRIVATE |
| No user messages | PRIVATE |

A prompt is only routed to a public LLM when the classifier is confident it
contains no sensitive content.

---

## Audit Log

Every routing decision is logged to `audit.jsonl`. Logs contain metadata only.
Prompt content is never written anywhere.

```json
{
  "request_id": "a3f9c2d1-...",
  "timestamp": "2026-09-12T14:32:11Z",
  "classification": "PRIVATE",
  "layer_triggered": "rule",
  "rule_matched": "ssn",
  "confidence": 1.0,
  "routed_to": "secret_ai",
  "model_requested": "auto",
  "latency_ms": {
    "classification": 0.5,
    "inference": 1200.0,
    "total": 1200.5
  }
}
```

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible routing endpoint |
| `/classify` | POST | Classification only — no LLM call |
| `/feedback` | POST | Save labeled feedback for retraining |
| `/health` | GET | Health check |
| `/` | GET | Test UI (Chat + Classifier tabs) |

### Test UI

The router serves a test UI at `/`:

- **Chat tab** — full conversation with live classification and routing.
  Shows which endpoint each message was routed to.
- **Classifier tab** — test DistilBERT on single inputs. Correct/Wrong buttons
  save labeled feedback to `feedback.jsonl` for future retraining.

---

## Configuration

All configuration via environment variables. Copy `.env.example` to `.env` to start.

| Variable | Default | Description |
|---|---|---|
| `SECRET_AI_ENDPOINT` | required | Secret AI TEE endpoint |
| `SECRET_AI_API_KEY` | required | Secret AI API key |
| `PUBLIC_LLM_ENDPOINT` | required | Public LLM endpoint |
| `PUBLIC_LLM_API_KEY` | required | Public LLM API key |
| `DISTILBERT_MODEL_PATH` | `training/model.onnx` | Path to ONNX classifier |
| `CONFIDENCE_THRESHOLD` | `0.85` | Below this → PRIVATE |
| `CLASSIFICATION_TIMEOUT_MS` | `2000` | Classifier timeout → PRIVATE |
| `DEFAULT_POLICY` | `private` | Fallback when uncertain |
| `PROXY_TIMEOUT_S` | `120` | LLM inference timeout |
| `AUDIT_LOG_FILE` | `audit.jsonl` | Audit log path |
| `SECRET_AI_DEFAULT_MODEL` | `qwen3:8b` | Fallback model for Secret AI |
| `PUBLIC_LLM_DEFAULT_MODEL` | `llama-3.3-70b-versatile` | Fallback model for public LLM |

---

## Training the Classifier

The DistilBERT classifier must be trained before the router can use it.
Training runs locally on your GPU and takes ~10-15 minutes.

```bash
# Install training dependencies
pip install -r training/requirements-training.txt

# Step 1 — Generate labeled training data via Claude (~30 min, uses Anthropic API)
ANTHROPIC_API_KEY=sk-... python training/generate_data.py
# Output: training/training_data.jsonl (~4,000 examples)

# Step 2 — Fine-tune DistilBERT (uses CUDA automatically)
python training/train.py
# Target: accuracy > 95%, false negative rate < 2%

# Step 3 — Export to ONNX for CPU inference
python training/export_onnx.py
# Output: training/model.onnx
```

### Retraining with Feedback

The test UI collects correct/incorrect labels from real usage. Incorporate them:

```bash
# Pull feedback from deployed instance
scp root@<your-secret-vm>:/app/feedback.jsonl ./feedback.jsonl

# Retrain (merges training_data.jsonl + feedback.jsonl automatically)
python training/train.py

# Re-export
python training/export_onnx.py

# Redeploy
docker compose build && docker compose up -d
```

---

## Deployment

SecretOrNot is designed to run inside a Secret VM (Intel TDX).

```bash
# On your Secret VM
git clone https://github.com/your-org/secretornot
cd secretornot
cp .env.example .env
# configure .env

docker compose up -d
```

### Exposing via Cloudflare (DNS Only)

SecretOrNot uses TEE attestation as its core trust guarantee. To preserve this,
Cloudflare must be set to DNS Only (grey cloud) — not proxied. Enabling the
Cloudflare proxy means Cloudflare terminates TLS and sees raw prompts before
they reach the TEE, which breaks the privacy guarantee.

In Cloudflare DNS for your domain:

```
Type:   A
Name:   son          (for son.attestai.io)
IPv4:   <your Secret VM public IP>
Proxy:  OFF (grey cloud — required)
TTL:    Auto
```

Access: `http://son.attestai.io:8000`

For HTTPS without Cloudflare proxying, run Nginx on the Secret VM:

```nginx
server {
    listen 443 ssl;
    server_name son.attestai.io;

    ssl_certificate /etc/letsencrypt/live/son.attestai.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/son.attestai.io/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;       # required for SSE streaming
        proxy_cache off;
        proxy_read_timeout 300s;
        chunked_transfer_encoding on;
    }
}
```

### Security Properties

- All classification happens inside the TEE — decisions are attested
- Private prompts never leave the secure environment
- DistilBERT runs in-process — no network hop for classification
- Audit logs record metadata only — prompt content is never written to disk
- Classifier always fails toward PRIVATE — never fails open

---

## Project Structure

```
secretornot/
├── main.py                  FastAPI app, routing endpoint
├── config.py                Environment variable management
├── models.py                Pydantic schemas (OpenAI-compatible)
├── classifier/
│   ├── rule_filter.py       Layer 1: regex + keyword rules
│   └── model_classifier.py  Layer 2: DistilBERT ONNX inference
├── router/
│   ├── decision.py          Routing logic + fail-closed policy
│   └── proxy.py             Request forwarding + SSE passthrough
├── audit/
│   └── logger.py            Metadata-only audit log
├── training/
│   ├── generate_data.py     Generate labeled training examples via Claude
│   ├── train.py             Fine-tune DistilBERT
│   └── export_onnx.py       Export to ONNX for CPU deployment
├── .env.example             Configuration template
└── requirements.txt
```

---

## Why Not Just Tag Prompts Manually?

Manual tagging requires every developer on every team to correctly identify
sensitive content in every prompt — consistently, under deadline, in production.
That's a policy problem masquerading as a technical one.

SecretOrNot makes the decision automatically, enforces it structurally at the
infrastructure level, and proves it cryptographically via TEE attestation.
The guarantee doesn't depend on developer discipline.

---

## Built On

- [Secret Network](https://scrt.network) — confidential compute via Secret VM (Intel TDX)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased) — lightweight transformer for classification
- [ONNX Runtime](https://onnxruntime.ai) — CPU inference in TEE
- [FastAPI](https://fastapi.tiangolo.com) — async Python web framework
- [secretvm-verify](https://github.com/scrtlabs/secretvm-verify) — TEE attestation verification
