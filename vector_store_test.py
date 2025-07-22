from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retriever.rag_embedding import MODEL_PATH, COLLECTION_NAME

# 初始化模型
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

# 创建 Qdrant 本地客户端
client = QdrantClient(path="./retriever/qdrant_local_data")  # 数据保存在本地文件夹
# 列出所有集合
collections = client.get_collections()
print("现有集合:", [col.name for col in collections.collections])
# '光伏发电系统接入配电网时如何进行防孤岛保护检测?'光伏发电系统接入配电网检测规程.pdf11页
# '电化学储能电站接入电网的额定能量如何进行测试?'电化学储能电站接入电网测试规程.pdf13页
# '风力发电机在电网中的谐波电压适应性如何测试?'风力发电机组%20电网适应性测试规程.pdf第12页
query_text = '光伏发电系统接入配电网时如何进行防孤岛保护检测?'
query_vector = model.encode(query_text, normalize_embeddings=True)
search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)

for point in search_result.points:
    print(f"ID: {point.id}")
    print(f"Chunk ID: {point.payload.get('chunk_id', '')}")
    print(f"Chapter: {point.payload.get('chapter', '')}")
    print(f"Article: {point.payload.get('article', '')}")
    print(f"Text: {point.payload.get('text', '').strip()}")
    print(f"Score (Qdrant internal): {point.score:.4f}")
    print(f"Source: {point.payload.get('source', '')}")
    print(f"Keywords: {point.payload.get('keywords', [])}")
    print("=" * 50)
