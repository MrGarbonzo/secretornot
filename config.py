import os

# Server
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "8000"))

# Destination endpoints
SECRET_AI_ENDPOINT = os.environ["SECRET_AI_ENDPOINT"]
SECRET_AI_API_KEY = os.environ["SECRET_AI_API_KEY"]
PUBLIC_LLM_ENDPOINT = os.environ["PUBLIC_LLM_ENDPOINT"]
PUBLIC_LLM_API_KEY = os.environ["PUBLIC_LLM_API_KEY"]

# Classifier (Layer 2) — Gemma POC via Secret AI
CLASSIFIER_ENDPOINT = os.environ.get("CLASSIFIER_ENDPOINT", SECRET_AI_ENDPOINT)
CLASSIFIER_API_KEY = os.environ.get("CLASSIFIER_API_KEY", SECRET_AI_API_KEY)
CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL", "gemma3:4b")

# Classifier backend: "gemma" (POC) or "distilbert" (production)
CLASSIFIER_BACKEND = os.environ.get("CLASSIFIER_BACKEND", "gemma")
DISTILBERT_MODEL_PATH = os.environ.get("DISTILBERT_MODEL_PATH", "training/model.onnx")

# Timeouts and thresholds
CLASSIFICATION_TIMEOUT_MS = int(os.environ.get("CLASSIFICATION_TIMEOUT_MS", "2000"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))
DEFAULT_POLICY = os.environ.get("DEFAULT_POLICY", "private")

# Proxy timeout for LLM inference calls (seconds)
PROXY_TIMEOUT_S = int(os.environ.get("PROXY_TIMEOUT_S", "120"))

# Audit
AUDIT_LOG_FILE = os.environ.get("AUDIT_LOG_FILE", "audit.jsonl")
