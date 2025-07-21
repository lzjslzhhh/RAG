# test_versions.py
import numpy as np
from transformers import __version__ as tf_version
from sentence_transformers import __version__ as st_version

print(f"✅ NumPy版本: {np.__version__}")
print(f"✅ Transformers版本: {tf_version}")
print(f"✅ Sentence-Transformers版本: {st_version}")

# 检查关键功能
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('moka-ai/m3e-base')
# print("✅ 模型加载成功")