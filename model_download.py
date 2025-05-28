import os
import ssl
import urllib3
from transformers import AutoTokenizer, AutoModel

# HTTPS 인증서 검증 우회 (주의: 개발 환경에서만!)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 모델 경로
model_path = "BAAI/bge-reranker-base"

print("🔄 HuggingFace에서 모델 다운로드 중 (SSL 인증서 무시 중)...")

# 모델 다운로드 (trust_remote_code=True는 일부 모델에 필요함)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# 저장 경로 지정
save_path = "./model/bge-reranker-v2-m3"
os.makedirs(save_path, exist_ok=True)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"✅ 다운로드 및 저장 완료: {save_path}")
