# SecretOrNot — Privacy Router
## Project Plan

**Tagline:** Route what needs to be private. Prove it was.

---

## Status

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — POC | ✅ COMPLETE | All files built, all hard rules enforced |
| Phase 2 — Production Classifier | ✅ COMPLETE | Code built, training ready to run on RTX 5070 Ti |
| Phase 3 — SDK Convenience Layer | 🔲 Not started | PrivacyChatSecret wrapper |
| Phase 4 — Platform Integration | 🔲 Not started | Pitch to Secret AI team |

---

## What Was Built (Phase 1)

```
secretornot/
├── main.py                      ← FastAPI app, POST /v1/chat/completions + GET /health
├── config.py                    ← All config from env vars, no hardcoded values
├── models.py                    ← Pydantic schemas (Classification, Destination, OpenAI request)
├── classifier/
│   ├── __init__.py
│   ├── rule_filter.py           ← Layer 1: 12 regex rules, short-circuits to PRIVATE
│   └── model_classifier.py      ← Layer 2: Gemma3-4B via Secret AI, swappable interface
├── router/
│   ├── __init__.py
│   ├── decision.py              ← Runs L1→L2, applies confidence threshold, fail-to-private
│   └── proxy.py                 ← Forward to destination, SSE streaming passthrough
├── audit/
│   ├── __init__.py
│   └── logger.py                ← JSONL metadata-only audit log, never writes prompts
├── training/
│   ├── generate_data.py         ← Phase 2 stub (now implemented)
│   ├── train.py                 ← Phase 2 stub (now implemented)
│   └── export_onnx.py           ← Phase 2 stub (now implemented)
├── requirements.txt
├── .env.example
└── README.md
```

### Phase 1 Design Decisions — Confirmed Delivered

1. **Swappable classifier** — model_classifier.py exposes a single async
   classify(text) → dict function. Swap the body for ONNX DistilBERT in
   Phase 2, nothing else changes.

2. **Layer 1 short-circuits** — rule_filter.check() returns on first regex
   match. If matched, decision.decide() returns PRIVATE immediately and never
   calls Layer 2.

3. **Audit never sees prompts** — logger.log_decision() takes only metadata
   fields. The function signature doesn't even accept a text parameter.

4. **Fail-closed everywhere** — Classifier timeout → PRIVATE. Classifier
   error → PRIVATE. Low confidence → PRIVATE. JSON parse failure → PRIVATE.
   The _fail_private() helper enforces this.

5. **Streaming works** — proxy.py uses httpx.AsyncClient.stream() with
   aiter_bytes() for SSE passthrough. Non-streaming uses standard
   POST → JSONResponse.

6. **Fully transparent** — Raw request body forwarded as-is. Response passed
   through untouched. Client sees a standard OpenAI response.

---

## What Was Built (Phase 2)

### training/generate_data.py ✅
- Calls Claude (Sonnet) via Anthropic SDK
- Generates 4,000 labeled examples: 2,000 PRIVATE, 2,000 PUBLIC
- PRIVATE covers 10 categories including ambiguous edge cases:
  implicit business sensitivity, HR/personnel, mixed prompts,
  personal medical queries without explicit keywords
- PUBLIC covers 9 categories including near-miss edge cases
- Generates in batches of 50 with progress reporting
- Output: training/training_data.jsonl
  Each line: {"text": "...", "label": 0} (PUBLIC) or {"label": 1} (PRIVATE)

### training/train.py ✅
- Fine-tunes distilbert-base-uncased
- 80/20 stratified train/eval split
- 3 epochs, early stopping (patience=3), best model by F1
- Auto-uses CUDA (RTX 5070 Ti)
- Prints: accuracy, precision, recall, F1, false negative rate
- Targets: accuracy > 95%, FNR < 2%
- Saves checkpoint to training/model_checkpoint/

### training/export_onnx.py ✅
- Loads checkpoint from training/model_checkpoint/
- Exports to training/model.onnx (opset 18)
- Validates: 10 samples through both PyTorch and ONNX — must match exactly
- Prints final model file size (target ~250MB)

### classifier/model_classifier.py ✅ (updated)
- Added CLASSIFIER_BACKEND env var (default: "gemma", alternative: "distilbert")
- DistilBERT path: loads model.onnx once at startup, runs via onnxruntime
  in thread executor (non-blocking async)
- Gemma path: existing behavior, completely unchanged
- Verified: Gemma backend still passes fail-closed test
- DistilBERT backend: 32-38ms on CPU (untrained model — will improve after
  fine-tuning on training data)
- async classify(text) → dict signature identical for both backends

### Phase 2 Remaining Action Items
Before the trained model is production-ready, these steps must be run locally:

1. Run generate_data.py — generates training_data.jsonl (~30 min with API calls)
2. Run train.py — fine-tunes on RTX 5070 Ti (~10-15 min)
3. Confirm metrics meet targets: accuracy > 95%, FNR < 2%
4. Run export_onnx.py — produces model.onnx
5. Set CLASSIFIER_BACKEND=distilbert and benchmark vs Gemma baseline
6. Deploy model.onnx to Secret VM alongside the router

---

## Architecture

```
Developer / Any LLM Client (OpenAI-compatible)
        ↓
┌─────────────────────────────────────────────┐
│         SecretOrNot Router (FastAPI)         │  ← Runs inside Secret VM / TEE
│                                             │
│  1. Request intake  POST /v1/chat/completions│
│  2. Layer 1: Rule-based pre-filter          │  ← regex/keywords (~1ms)
│  3. Layer 2: Model classifier               │  ← DistilBERT ONNX (~35ms) 
│  4. Routing decision                        │  ← default-to-private policy
│  5. Audit log (metadata only, no prompts)   │
└─────────────────────────────────────────────┘
          ↓                    ↓
     Secret AI             Public LLM
  (private prompts)      (OpenAI, etc.)
```

---

## Classifier Strategy

### POC: Gemma3-4B via Secret AI ✅
- Zero training required
- Constrained prompt: respond ONLY with PRIVATE or PUBLIC + confidence in JSON
- Latency: ~300-800ms
- Runs inside TEE — classification never leaves secure environment
- Still available via CLASSIFIER_BACKEND=gemma

### Production: Fine-tuned DistilBERT on CPU ✅ (code complete, training pending)
- 66M parameter binary classifier
- 32-38ms on CPU (untrained) — will be faster after fine-tuning
- Fits entirely in TEE memory (~250MB model, ~500MB RAM)
- Deterministic — same input always produces same output
- No network dependency — runs in-process
- Enabled via CLASSIFIER_BACKEND=distilbert

### Local Training Environment
- Hardware: RTX 5070 Ti (16GB GDDR7, 1406 AI TOPs, Blackwell architecture)
- Training data: 4,000 labeled examples generated with Claude
- Framework: Hugging Face transformers
- Time to train: ~10-15 minutes
- Export format: ONNX (opset 18, for CPU inference in TEE)

---

## Hard Rules (Non-Negotiable)

1. Never log prompt content — metadata only
2. Always fail toward PRIVATE on timeout or error
3. Response format must be identical to standard OpenAI response
4. Streaming must work via SSE passthrough
5. All config via environment variables — no hardcoded values
6. Classification decision is made before any data leaves the TEE

---

## Security Properties

- Router runs inside Secret VM (TEE) — all classification is attested
- Routing decision itself can be logged on-chain as attestation record
- Private prompts never touch the public LLM endpoint
- No prompt content ever written to disk or logs
- Classifier timeout defaults to PRIVATE — never fails open
- DistilBERT runs in-process — no network hop for classification

---

## Deployment Model

### Self-Hosted (Current / OSS)
Each developer runs their own instance in their own Secret VM.
Full data isolation — no prompts touch external infrastructure.
You ship the code, they operate it.

### Hosted SaaS (Future)
You run a shared router inside Secret VM.
Developers get an API key from SecretOrNot.
Integration: change base_url + api_key only.
TEE attestation is the trust argument — the TEE proves correct handling
cryptographically, no need to ask customers to trust you.

---

## Environment Variables

```
ROUTER_PORT=8000
SECRET_AI_ENDPOINT=https://...
SECRET_AI_API_KEY=...
PUBLIC_LLM_ENDPOINT=https://api.openai.com
PUBLIC_LLM_API_KEY=...
CLASSIFIER_BACKEND=distilbert        # "gemma" or "distilbert"
CLASSIFIER_ENDPOINT=...              # Gemma only: Secret AI endpoint
CLASSIFIER_API_KEY=...               # Gemma only
CLASSIFIER_MODEL=gemma3:4b           # Gemma only
CLASSIFICATION_TIMEOUT_MS=100
DEFAULT_POLICY=private
CONFIDENCE_THRESHOLD=0.85
PROXY_TIMEOUT_S=120
AUDIT_LOG_FILE=audit.jsonl
```

---

## Roadmap

### Phase 1 — POC ✅ COMPLETE
Standalone router with Gemma3-4B classifier.
Routing concept proven. Demo-ready.

### Phase 2 — Production Classifier ✅ COMPLETE (training pending)
All code built and validated. DistilBERT ONNX backend running at 32-38ms.
Remaining: run training pipeline locally on RTX 5070 Ti, deploy model.onnx.

### Phase 3 — SDK Convenience Layer 🔲
Thin PrivacyChatSecret wrapper that auto-routes to the standalone router.
Developer experience: one import, same interface as before.

### Phase 4 — Platform Integration 🔲
Pitch SDK-native router to Secret AI team.
Classifier lives in Secret AI infrastructure.
Every SDK user gets privacy routing automatically.

---

## Why This Matters

Most "privacy-preserving" LLM solutions require developers to manually decide
what to keep private. SecretOrNot makes that decision automatically, enforces
it structurally, and proves it cryptographically via TEE attestation.

The pitch: "Your developers don't have to think about what's private.
The router does. And you can prove it."
