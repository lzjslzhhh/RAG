import json
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 配置
COLLECTION_NAME = "grid_docs"
MODEL_PATH = r"/tmp/pycharm_project_581/gte-multilingual-base"
# JSONL_PATH = r"/tmp/pycharm_project_581/retriever/rag_enhanced_2_chunks.jsonl"
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线
if __name__ == "__main__":
    # 初始化模型
    model = SentenceTransformer(model_name_or_path=MODEL_PATH, trust_remote_code=True, local_files_only=True)

    documents = []
    dir_path = '/tmp/pycharm_project_581/EleQA-master/documents'
    for root,_,files in os.walk(dir_path):
        for file in tqdm(files, desc="Processing files"):
            if not file.endswith(('.json', '.jsonl')):
                continue

            file_path = os.path.join(root, file)
            print(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for chunk in data:
                        # documents.append({
                        #     "id": doc["chunk_id"],
                        #     "text": enriched_text,
                        #     "raw": doc["content"],
                        #     "metadata": doc["metadata"],
                        #     "chapter": doc.get("chapter_title", ""),
                        #     "article": doc.get("article_no", ""),
                        #     "source": doc.get("source", "")
                        # })
                        documents.append({
                            "chunk_id": chunk["chunk_id"],
                            'doc_id':chunk["doc_id"],
                            'level': chunk["level"],
                            "text": chunk["chunk_content"],
                            "source": chunk["doc_name"],
                            "path": chunk["path"]
                        })
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    # 创建 Qdrant 本地客户端
    client = QdrantClient(path="../qdrant_local_data")  # 数据保存在本地文件夹

    # 创建集合
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    # 批量生成向量并写入
    points = []
    for i, doc in enumerate(tqdm(documents)):
        # embedding = model.encode(f'{doc["source"]} {doc["text"]}', normalize_embeddings=True).tolist()
        # embedding = model.encode(f'{doc["source"]} {doc["title"]} {doc["text"]}', normalize_embeddings=True).tolist()
        # 分别编码后加权融合
        source_emb = model.encode(doc["source"], weight=0.2, normalize_embeddings=True)
        context_emb = model.encode(doc['chunk_id'] + doc["path"], weight=0.25,
                                   normalize_embeddings=True)
        content_emb = model.encode(doc["text"], weight=0.55, normalize_embeddings=True)
        combined_emb = source_emb * 0.2 + context_emb * 0.25 + content_emb * 0.55
        # combined_emb = source_emb * 0.3 + content_emb * 0.7
        # point_id =  f"{doc['source']}_{doc['chapter']}_{doc['id']}"
        point_id = f"{doc['source']}_{doc['chunk_id']}"
        # 查询是否已经存在这个 ID
        existing = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id]
        )
        if existing:  # 已存在就跳过
            continue
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id))  # 保持相同ID生成相同UUID
        points.append(PointStruct(
            id=str(point_id),
            vector=combined_emb,
            payload={
                "chunk_id": doc["chunk_id"],
                "doc_id": doc["doc_id"],
                "level": doc["level"],
                "text": doc["source"] +' '+ doc["chunk_id"] + ' '+ doc["text"],
                "source": doc["source"],
                "hierarchy": doc["path"],
            },
            # payload = {
            #     "chunk_id": doc["id"],
            #     "chapter": doc["chapter"],
            #     "article": doc["article"],
            #     "text": doc["raw"],
            #     "source": doc["source"],
            #     "keywords": doc["metadata"].get("keywords", [])
            # }
        ))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
