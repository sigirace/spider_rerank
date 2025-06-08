model repo

- https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main

pip install -r requirements.txt

python downloads.py


```
docker buildx build --platform linux/amd64,linux/arm64 \
  -t sigirace/rerank:latest \
  . \
  --push
```