from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings.base import Embeddings  # RAGAS 所需接口

class GTEEmbedding(Embeddings):
    def __init__(self, model_name_or_path: str = r"/tmp/pycharm_project_581/gte-multilingual-base", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token
            embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize to unit vector
        return embeddings[0].cpu().tolist()
