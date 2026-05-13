"""
Embedding layer: Convert incident text to vectors.

Why Mistral embeddings?
- 768-dimensional vectors capture semantic meaning better than smaller models
- Open-source and runs locally (no API dependency)
- Good for incident similarity: semantically similar incidents cluster together

Interview talking point:
"We use Mistral's SFR embeddings because precision matters in incident analysis.
The 768-dim vectors allow fine-grained similarity comparisons between incidents,
which translates to better Precision@5 in retrieval."
"""

from sentence_transformers import SentenceTransformer
from src.schemas import Incident, IncidentEmbedding


class IncidentEmbedder:
    """Embeds incidents using Mistral SFR model."""

    MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"  # Mistral SFR equivalent
    EMBEDDING_DIM = 1024

    def __init__(self):
        """Load the embedding model (downloads on first run)."""
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed_incident(self, incident: Incident) -> IncidentEmbedding:
        """
        Convert an incident to an embedding.

        Why embed both title AND description?
        - Title captures the incident type ("Database exhaustion")
        - Description captures context and nuance ("unclosed connections in v2.3.1")
        - Together they form a complete semantic representation
        """

        # Combine title + description for richer context
        text_to_embed = f"Title: {incident.title}\nDescription: {incident.description}"

        # Generate the embedding (768-dim vector)
        embedding = self.model.encode(
            text_to_embed,
            convert_to_tensor=False,
        ).tolist()

        return IncidentEmbedding(
            incident=incident,
            embedding=embedding,
            embedding_text=text_to_embed,
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query (e.g., "database connection issues").

        Why separate method?
        - Queries use the same embedding space as incidents
        - This allows semantic search: query vector vs incident vectors
        """
        return self.model.encode(query, convert_to_tensor=False).tolist()

    def search_similar(
        self,
        query: str,
        incident_embeddings: list[IncidentEmbedding],
        top_k: int = 5,
    ) -> list[tuple[Incident, float]]:
        """
        Find similar incidents to a query using cosine similarity.

        Why not just use Weaviate for this?
        - This is for local/testing
        - Weaviate will do the real retrieval at scale
        - This shows the concept before introducing the vector DB
        """
        from numpy import dot
        from numpy.linalg import norm

        query_embedding = self.embed_query(query)

        # Calculate cosine similarity with each incident
        similarities = []
        for inc_emb in incident_embeddings:
            # Cosine similarity = dot product / (norm * norm)
            similarity = dot(query_embedding, inc_emb.embedding) / (
                norm(query_embedding) * norm(inc_emb.embedding)
            )
            similarities.append((inc_emb.incident, similarity))

        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
