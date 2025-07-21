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
query_text = "供电用户不得有哪些行为"
query_vector = model.encode(query_text, normalize_embeddings=True)
search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=10
)

# Step 2: 计算每个结果与 query 的相似度，并排序
scored_points = []
for point in search_result.points:
    text = point.payload.get('text', '').strip()
    embedding = model.encode(text, normalize_embeddings=True)
    similarity = model.similarity(query_vector, embedding)
    scored_points.append((similarity, point))

# Step 3: 根据相似度排序，取前 3 个
top_points = sorted(scored_points, key=lambda x: x[0], reverse=True)[:3]

# 查看search_result有哪些属性，调试用：
print(dir(search_result))


# Step 4: 打印结果
for sim, point in top_points:
    print(f"ID: {point.id}")
    print(f"Chunk ID: {point.payload.get('chunk_id', '')}")
    print(f"Chapter: {point.payload.get('chapter', '')}")
    print(f"Article: {point.payload.get('article', '')}")
    print(f"Text: {point.payload.get('text', '').strip()}")
    print(f"Similarity: {sim.item():.4f}")
    print(f"Score (Qdrant internal): {point.score:.4f}")
    print(f"Source: {point.payload.get('source', '')}")
    print(f"Keywords: {point.payload.get('keywords', [])}")
    print("=" * 50)