import torch
import math
from typing import List
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": "bfloat16"},
            tokenizer_kwargs={"padding_side": "left"}
        )

    def get_emb(self, input, query_type=False) -> torch.Tensor:
        if query_type:
            embeddings = self.model.encode([input], prompt_name="query")
        else:
            embeddings = self.model.encode(input)
        return embeddings

    def soft_retrieve_eval(self, query: str, documents: List[str], gth: List) -> torch.Tensor:
        query_embedding = self.model.encode([query], prompt_name="query")
        document_embeddings = self.model.encode(documents)
        similarity_scores = self.model.similarity(query_embedding, document_embeddings)
        
        positive_score = 1.0
        negative_score = 0.0
        for id in len(similarity_scores):
            if id in gth:
                positive_score += similarity_scores[id].item()
            else:
                negative_score += similarity_scores[id].item()
        positive_score /= len(gth) + 1
        negative_score /= len(similarity_scores) - len(gth) + 1

        contrastive_score = positive_score - negative_score
        
        return contrastive_score


    def retrieve(self, query: str, documents: List[str], top_k: int = 10) -> torch.Tensor:
        """
        Args:
            query (str): A query string to be encoded.
            documents (List[str]): A list of document strings to be encoded.
            top_k (int, optional): Returns the top k most similar documents for each query.
        Returns:
            similarity_scores (torch.Tensor): A tensor containing the similarity scores between the queries and documents.
        """
        query_embedding = self.model.encode([query], prompt_name="query")
        document_embeddings = self.model.encode(documents)
        similarity_scores = self.model.similarity(query_embedding, document_embeddings)
        indices = torch.topk(similarity_scores, min(top_k, similarity_scores.size(1))).indices
        indices = indices.squeeze(0)
        return indices
    
    def eval(self, indices: torch.Tensor, gth: List) -> tuple[float, float]:
        """
        args:
            indices (torch.Tensor): A tensor containing the indices of the retrieved documents, size: 1xm. 
            gth (List): A tensor containing the ground truth indices for evaluation, size: 1xn.
        """
        indices = list(dict.fromkeys(indices.tolist()).keys())
        gth = list(dict.fromkeys(gth).keys())
        joint = set(indices).intersection(gth)
        precision = len(joint) / len(indices) if indices else 0.0
        recall = len(joint) / len(gth) if gth else 0.0
        if len(gth) == 0:
            ndcg = 1.0
        else:
            dcg, idcg = 0.0, 0.0
            for i in range(len(indices)):
                if indices[i] in gth:
                    dcg += 1 / math.log2(i + 2)
            for i in range(len(gth)):
                if i == len(indices):
                    break
                idcg += 1 / math.log2(i + 2)
            ndcg = dcg / idcg

        return precision, recall, ndcg