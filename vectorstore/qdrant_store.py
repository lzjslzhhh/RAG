from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from retriever.rag_embedding import MODEL_PATH, COLLECTION_NAME


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

def load_qdrant_vectorstore(collection_name=COLLECTION_NAME, model_name=MODEL_PATH):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = SentenceTransformerEmbeddings(model)

    # 创建 Qdrant 本地客户端
    client = QdrantClient(path="./qdrant_local_data")  # 数据保存在本地文件夹

    # 创建集合（如果不存在）
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    return vectorstore