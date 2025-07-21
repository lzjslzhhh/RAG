import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# 配置
COLLECTION_NAME = "grid_docs"
MODEL_PATH = "Alibaba-NLP/gte-multilingual-base"
JSONL_PATH = "/tmp/pycharm_project_581/retriever/rag_structured_chunks.jsonl"

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
            keywords = " ".join(doc.get("metadata", {}).get("keywords", []))
            enriched_text = f"{base_text} 关键词：{keywords}"
            if doc['section'] == '正文':
                documents.append({
                    "id": doc["chunk_id"],
                    "text": enriched_text,
                    "raw": doc["content"],
                    "metadata": doc["metadata"],
                    "chapter": doc.get("chapter_title", ""),
                    "article": doc.get("article_no", ""),
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
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "chunk_id": doc["id"],
                "chapter": doc["chapter"],
                "article": doc["article"],
                "text": doc["raw"],
                "source": doc["source"],
                "keywords": doc["metadata"].get("keywords", [])
            }
        ))

    # print(points)
    client.upsert(collection_name=COLLECTION_NAME, points=points)


