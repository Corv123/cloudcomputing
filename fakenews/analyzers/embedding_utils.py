"""
Embedding utilities for generating and comparing article embeddings.
Used by both Spark pipeline (batch) and Lambda (runtime).
"""

from typing import List, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


# Global model instance for Lambda (singleton)
_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """
    Get or create embedding model instance (singleton pattern).
    
    Args:
        model_name: Name of the sentence-transformer model
    
    Returns:
        SentenceTransformer model instance, or None if not available
    """
    global _embedding_model
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            return None
    
    return _embedding_model


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
    """
    Generate embedding vector for a text string.
    
    Args:
        text: Text to embed (should be article content)
        model_name: Name of the sentence-transformer model
    
    Returns:
        List of floats representing the embedding vector (384 dimensions for all-MiniLM-L6-v2),
        or None if error
    """
    if not text or len(text.strip()) < 10:
        return None
    
    model = get_embedding_model(model_name)
    if model is None:
        return None
    
    try:
        # Generate embedding
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Convert to list of floats
        return embedding.tolist()
    
    except Exception as e:
        print(f"[WARN] Error generating embedding: {e}")
        return None


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score between -1 and 1 (higher = more similar)
    """
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return float(similarity)
    
    except Exception as e:
        print(f"[WARN] Error calculating cosine similarity: {e}")
        return 0.0


def find_top_similar(
    query_embedding: List[float],
    article_embeddings: List[Tuple[dict, List[float]]],
    top_k: int = 10
) -> List[Tuple[dict, float]]:
    """
    Find top-k most similar articles using cosine similarity.
    
    Args:
        query_embedding: Embedding of the query article
        article_embeddings: List of (article_dict, embedding) tuples
        top_k: Number of top results to return
    
    Returns:
        List of (article_dict, similarity_score) tuples, sorted by similarity (highest first)
    """
    similarities = []
    
    for article_dict, article_embedding in article_embeddings:
        if article_embedding is None:
            continue
        
        sim = cosine_similarity(query_embedding, article_embedding)
        similarities.append((article_dict, sim))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return similarities[:top_k]


def aggregate_similarity_scores(
    similarities: List[Tuple[dict, float]],
    method: str = "max"
) -> float:
    """
    Aggregate similarity scores using various methods.
    
    Args:
        similarities: List of (article_dict, similarity_score) tuples
        method: Aggregation method ('max', 'mean', 'weighted_mean', 'top_k_mean')
    
    Returns:
        Aggregated similarity score
    """
    if not similarities:
        return 0.0
    
    scores = [sim for _, sim in similarities]
    
    if method == "max":
        return max(scores)
    
    elif method == "mean":
        return sum(scores) / len(scores)
    
    elif method == "weighted_mean":
        # Weight by source reliability
        total_weight = 0.0
        weighted_sum = 0.0
        
        for article_dict, sim in similarities:
            weight = 1.5 if article_dict.get('is_reliable', False) else 1.0
            weighted_sum += sim * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    elif method == "top_k_mean":
        # Average of top 5
        k = min(5, len(scores))
        top_scores = sorted(scores, reverse=True)[:k]
        return sum(top_scores) / len(top_scores)
    
    else:
        return max(scores)  # Default to max


