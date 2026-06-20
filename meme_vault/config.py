import os

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
API_BASE = "https://api.siliconflow.cn/v1"

EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

TOP_K = 5
RETRY_COUNT = 3
