from sentence_transformers import SentenceTransformer

class MyEmbedder:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
