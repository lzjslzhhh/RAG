import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# 配置
COLLECTION_NAME = "grid_docs"
MODEL_PATH = "D:\learning\RAG\RAG\gte-multilingual-base"
JSONL_PATH = r"D:\learning\RAG\RAG\retriever\rag_hierarchy_chunks.jsonl"

if __name__ == "__main__":
    # 初始化模型
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

    # 加载文档并拼接关键词
    documents = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            base_text = doc["content"].strip().replace("\n", "")
            # 安全地获取 keywords
            # keywords = " ".join(doc.get("metadata", {}).get("keywords", []))
            # enriched_text = f"{base_text} 关键词：{keywords}"
            if doc['section'] == '正文':
                documents.append({
                    "id": doc["chunk_id"],
                    'title': doc["title"],
                    'level': doc["level"],
                    "text": base_text,
                    "source": doc.get("source", "")
                })

    # 创建 Qdrant 本地客户端
    client = QdrantClient(path="./qdrant_local_data")  # 数据保存在本地文件夹

    # 创建集合（如果不存在）
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    # 批量生成向量并写入
    points = []
    for i, doc in enumerate(tqdm(documents)):
        embedding = model.encode(doc["text"], normalize_embeddings=True).tolist()

        point_id = doc["id"]

        # 查询是否已经存在这个 ID（可选优化：用批量方式）
        existing = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )
        if existing:  # 已存在就跳过
            continue

        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk_id": doc["id"],
                "title": doc["title"],
                "level": doc["level"],
                "text": doc["text"],
                "source": doc["source"],
            }
        ))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)


