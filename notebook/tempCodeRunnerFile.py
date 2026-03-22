import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")

print("Persist directory:", DEFAULT_PERSIST_DIR)
print("Absolute path:", os.path.abspath(DEFAULT_PERSIST_DIR))