import os

# Server
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "8000"))

# Destination endpoints
SECRET_AI_ENDPOINT = os.environ["SECRET_AI_ENDPOINT"]
SECRET_AI_API_KEY = os.environ["SECRET_AI_API_KEY"]
PUBLIC_LLM_ENDPOINT = os.environ["PUBLIC_LLM_ENDPOINT"]
PUBLIC_LLM_API_KEY = os.environ["PUBLIC_LLM_API_KEY"]

# Classifier (Layer 2) — fine-tuned DistilBERT via ONNX
DISTILBERT_MODEL_PATH = os.environ.get("DISTILBERT_MODEL_PATH", "training/model.onnx")

# Timeouts and thresholds
CLASSIFICATION_TIMEOUT_MS = int(os.environ.get("CLASSIFICATION_TIMEOUT_MS", "2000"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))
DEFAULT_POLICY = os.environ.get("DEFAULT_POLICY", "private")

# Default models per destination — used when the requested model doesn't exist on the target
SECRET_AI_DEFAULT_MODEL = os.environ.get("SECRET_AI_DEFAULT_MODEL", "qwen3:8b")
PUBLIC_LLM_DEFAULT_MODEL = os.environ.get("PUBLIC_LLM_DEFAULT_MODEL", "llama-3.3-70b-versatile")

# Proxy timeout for LLM inference calls (seconds)
PROXY_TIMEOUT_S = int(os.environ.get("PROXY_TIMEOUT_S", "120"))

# Attestation — VM URLs for secretvm-verify SDK
# SECRET_AI_VM_URL defaults to the hostname from SECRET_AI_ENDPOINT
# SELF_VM_URL defaults to localhost (auto-detected at verification time)
def _default_secret_ai_vm_url() -> str:
    from urllib.parse import urlparse
    parsed = urlparse(SECRET_AI_ENDPOINT)
    host = parsed.hostname or SECRET_AI_ENDPOINT
    return f"https://{host}"

SECRET_AI_VM_URL = os.environ.get("SECRET_AI_VM_URL") or _default_secret_ai_vm_url()
SELF_VM_URL = os.environ.get("SELF_VM_URL", "https://localhost")

# Audit
AUDIT_LOG_FILE = os.environ.get("AUDIT_LOG_FILE", "audit.jsonl")
