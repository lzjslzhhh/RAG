from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from retriever.rag_embedding import MODEL_PATH, COLLECTION_NAME


# 初始化模型
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

# 创建 Qdrant 本地客户端
client = QdrantClient(path="./qdrant_local_data")  # 数据保存在本地文件夹

query_text = "电力监控系统的安全保护措施"
query_vector = model.encode(query_text, normalize_embeddings=True)
search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3
)
# 查看search_result有哪些属性，调试用：
print(dir(search_result))


points = search_result.points
for point in points:
    print(f"ID: {point.id}")
    print(f"Chunk ID: {point.payload.get('chunk_id', '')}")
    print(f"Chapter: {point.payload.get('chapter', '')}")
    print(f"Article: {point.payload.get('article', '')}")
    print(f"Text: {point.payload.get('text', '').strip()}")
    embeddings = model.encode(point.payload.get('text', '').strip(),normalize_embeddings=True)
    print(f'{model.similarity(query_vector, embeddings)}')
    print(f"Score: {point.score:.4f}")
    print(f"Source: {point.payload.get('source', '')}")
    print(f"Keywords: {point.payload.get('keywords', [])}")
    print("=" * 50)