import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
IMAGES_DIR = os.path.join(ROOT, "images")
PUBLIC_DIR = os.path.join(os.path.dirname(ROOT), "public")

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
API_BASE = "https://api.siliconflow.cn/v1"

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

TOP_K = 5
BATCH_SIZE = 32
RETRY_COUNT = 3

RATE_LIMIT_RPM = 1000
RATE_LIMIT_TPM = 80000
