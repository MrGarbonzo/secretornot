FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py config.py models.py ./
COPY classifier/ classifier/
COPY router/ router/
COPY audit/ audit/
COPY training/model.onnx training/model.onnx
COPY training/model.onnx.data training/model.onnx.data
COPY training/model_checkpoint/tokenizer.json training/model_checkpoint/tokenizer.json
COPY training/model_checkpoint/tokenizer_config.json training/model_checkpoint/tokenizer_config.json

ENV DISTILBERT_MODEL_PATH=training/model.onnx

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
