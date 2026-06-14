import os
import pickle
from sentence_transformers import CrossEncoder

from sympy import vectorize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
import torch
import numpy as np



class Reranker:
    """
    Perform reranking of documents based on their relevance to a given query.

    The reranking strategy is selected at initialization via the `type` parameter.
    Depending on the strategy, different internal models and vectorizers are used.

    Attributes:
        type (str): Reranking strategy ('cross_encoder', 'tfidf', 'bow', 'hybrid', 'sequential').
        cross_encoder_model_name (str): Pretrained model name for cross-encoder.
        cross_encoder_model: Loaded transformer model (if applicable).
        tokenizer: Tokenizer for the cross-encoder model.
    """
    def __init__(self, type, cross_encoder_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2', corpus_directory=''):
        """
        Initialize the Reranker with a given strategy and optional cross-encoder model.

        Args:
            type (str): Reranking strategy. Must be one of: 'cross_encoder', 'tfidf',
                        'bow', 'hybrid', 'sequential'.
            cross_encoder_model_name (str, optional): HuggingFace model identifier for the
                                                      cross-encoder. Defaults to MiniLM.
            corpus_directory (str, optional): Directory for corpus files (currently unused).
        """
        self.type = type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_model = None
        self.tokenizer = None

        if self.type in ["cross_encoder", "hybrid", "sequential"]:
            self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)

    def rerank(self, query, context, distance_metric="cosine", seq_k1=None, seq_k2=None):
        """
        Dispatch reranking to the appropriate method based on the strategy type.

        Args:
            query (str): The search query.
            context (List[str]): List of document strings to rerank.
            distance_metric (str, optional): Distance metric for TF-IDF/BoW (e.g., 'cosine', 'euclidean').
                                             Defaults to 'cosine'.
            seq_k1 (int, optional): Number of documents to keep after the first stage (TF-IDF) in sequential reranking.
            seq_k2 (int, optional): Number of documents to keep after the second stage (cross-encoder) in sequential reranking.

        Returns:
            tuple: (ranked_docs, ranked_indices, ranked_scores)
                - ranked_docs: List of documents in descending relevance order.
                - ranked_indices: Original indices of the documents in the ranked order.
                - ranked_scores: Relevance scores (or distances, depending on the strategy).

        Raises:
            ValueError: If an unsupported reranking type is provided.
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "bow":
            return self.bow_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "sequential":
            return self.sequential_rerank(query, context, seq_k1, seq_k2, distance_metric=distance_metric)
        else:
            raise ValueError(f"Unsupported reranking strategy: {self.type}")

    def cross_encoder_rerank(self, query, context):
        """
        Rerank documents using a cross-encoder transformer model.

        The cross-encoder processes (query, document) pairs simultaneously and outputs
        a relevance logit. Higher logits indicate higher relevance.

        Args:
            query (str): The search query.
            context (List[str]): List of document strings.

        Returns:
            tuple: (ranked_docs, ranked_indices, ranked_scores)
                - ranked_docs: Documents sorted by descending relevance.
                - ranked_indices: Original indices in that order.
                - ranked_scores: Raw logit scores.
        """
        if not context:
            return [], [], []

        if self.cross_encoder_model is None or self.tokenizer is None:
            raise RuntimeError(
                f"Cross-encoder model '{self.cross_encoder_model_name}' is not loaded."
            )

        query_document_pairs = [(query, document) for document in context]

        inputs = self.tokenizer(
            query_document_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits.squeeze(-1)
            relevance_scores = logits.detach().cpu().numpy().astype(float)

        ranked_indices = np.argsort(-relevance_scores)
        ranked_docs = [context[index] for index in ranked_indices]
        ranked_scores = [float(relevance_scores[index]) for index in ranked_indices]

        return ranked_docs, ranked_indices.tolist(), ranked_scores

    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using TF-IDF vectorization and distance-based similarity.
        """
        if not context:
            return [], [], []

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([query] + context)
        query_vector = matrix[0]
        context_vectors = matrix[1:]

        distances = pairwise_distances(query_vector, context_vectors, metric=distance_metric).flatten()
        ranked_indices = np.argsort(distances)
        ranked_docs = [context[index] for index in ranked_indices]
        ranked_distances = [float(distances[index]) for index in ranked_indices]
        return ranked_docs, ranked_indices.tolist(), ranked_distances

    def bow_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using BoW vectorization and distance-based similarity.
        """
        if not context:
            return [], [], []

        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform([query] + context)
        query_vector = matrix[0]
        context_vectors = matrix[1:]

        distances = pairwise_distances(query_vector, context_vectors, metric=distance_metric).flatten()
        ranked_indices = np.argsort(distances)
        ranked_docs = [context[index] for index in ranked_indices]
        ranked_distances = [float(distances[index]) for index in ranked_indices]
        return ranked_docs, ranked_indices.tolist(), ranked_distances

    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        if not context:
            return [], [], []

        _, _, tfidf_distances = self.tfidf_rerank(query, context, distance_metric=distance_metric)
        tfidf_similarity = 1 - np.array(tfidf_distances, dtype=float)

        _, _, cross_scores = self.cross_encoder_rerank(query, context)
        cross_scores = np.array(cross_scores, dtype=float)

        if cross_scores.ptp() > 0:
            cross_scores = (cross_scores - cross_scores.min()) / cross_scores.ptp()
        else:
            cross_scores = np.ones_like(cross_scores)

        combined_scores = tfidf_weight * tfidf_similarity + (1 - tfidf_weight) * cross_scores
        ranked_indices = np.argsort(-combined_scores)
        ranked_docs = [context[index] for index in ranked_indices]
        ranked_scores = [float(combined_scores[index]) for index in ranked_indices]
        return ranked_docs, ranked_indices.tolist(), ranked_scores

    def sequential_rerank(self, query, context, seq_k1, seq_k2, distance_metric="cosine"):
        """
        Apply a two-stage reranking pipeline: TF-IDF followed by cross-encoder.
        """
        if not context:
            return [], [], []

        if seq_k1 is None:
            seq_k1 = len(context)
        if seq_k2 is None:
            seq_k2 = seq_k1

        _, first_stage_indices, _ = self.tfidf_rerank(query, context, distance_metric=distance_metric)
        first_stage_indices = first_stage_indices[:seq_k1]
        first_stage_context = [context[index] for index in first_stage_indices]

        second_stage_docs, second_stage_indices, second_stage_scores = self.cross_encoder_rerank(query, first_stage_context)
        second_stage_docs = second_stage_docs[:seq_k2]
        second_stage_indices = second_stage_indices[:seq_k2]
        second_stage_scores = second_stage_scores[:seq_k2]

        final_indices = [first_stage_indices[index] for index in second_stage_indices]
        return second_stage_docs, final_indices, second_stage_scores


if __name__ == "__main__":
    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants that may help prevent cardiovascular disease.",
        "Coffee is also rich in antioxidants but can increase heart rate.",
        "Drinking water is essential for hydration.",
        "Green tea may also aid in weight loss and improve brain function."
    ]

    print("\nTF-IDF Reranking:")
    reranker = Reranker(type="tfidf")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nCross-Encoder Reranking:")
    reranker = Reranker(type="cross_encoder")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nHybrid Reranking:")
    reranker = Reranker(type="hybrid")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")
