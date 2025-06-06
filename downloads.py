from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="./model/bge-reranker-v2-m3",
    local_dir_use_symlinks=False,  # 파일 복사로 저장 (Docker 등에 유리)
)
