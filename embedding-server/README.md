# Local CPU embedding service

OpenAI-compatible `POST /v1/embeddings` service for novel_reader and NovelIQ.
It intentionally uses `CPUExecutionProvider` and caps concurrent inference at 8.

## Run

```bash
cd /home/oraw/projects/00.Program/novel_reader
.embedding-venv/bin/python -m uvicorn main:app \
  --app-dir embedding-server \
  --host 127.0.0.1 \
  --port 8001
```

The first start downloads `BAAI/bge-small-zh-v1.5` to `~/.cache/fastembed` unless
`FASTEMBED_CACHE_DIR` is set.

## Smoke test

```bash
curl -fsS http://127.0.0.1:8001/health | jq .

curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-small-zh-v1.5","input":["测试文本","第二条"],"encoding_format":"float"}' \
  | jq '{model, usage, dimensions: [.data[].embedding | length], indexes: [.data[].index]}'
```
