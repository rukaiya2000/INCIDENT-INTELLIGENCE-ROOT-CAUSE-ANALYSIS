"""
Data schemas for incidents.

Why Pydantic?
- Enforces type safety and validation at boundaries
- Makes it easy to convert to/from JSON for storage
- Interview talking point: "We use Pydantic because incident data must be reliable—
  wrong types break the pipeline. Validation at the edge prevents garbage data from
  entering the system."
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Incident(BaseModel):
    """Core incident data structure."""

    id: str = Field(..., description="Unique incident identifier")
    title: str = Field(..., description="Short incident title")
    description: str = Field(..., description="Detailed incident description (what happened)")
    timestamp: datetime = Field(..., description="When the incident occurred")

    # Metadata for filtering and context
    service: str = Field(..., description="Which service/component failed (e.g., 'auth-service', 'db-cluster')")
    environment: str = Field(default="production", description="Environment: production, staging, dev")
    severity: str = Field(..., description="Critical, High, Medium, Low")

    # Resolution info (filled after incident is resolved)
    root_cause: Optional[str] = Field(None, description="Identified root cause")
    resolution_steps: Optional[list[str]] = Field(None, description="How it was fixed")
    time_to_resolve: Optional[int] = Field(None, description="Minutes from detection to resolution")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "INC-2024-001",
                "title": "Database connection pool exhaustion",
                "description": "Auth service unable to connect to database. Connection pool hit max size due to unclosed connections in recent deployment.",
                "timestamp": "2024-05-13T14:30:00",
                "service": "auth-service",
                "environment": "production",
                "severity": "Critical",
                "root_cause": "Memory leak in connection handler from v2.3.1 deployment",
                "resolution_steps": ["Rolled back to v2.3.0", "Restarted service", "Monitored for 30 min"],
                "time_to_resolve": 15
            }
        }


class IncidentEmbedding(BaseModel):
    """Incident + its embedding, ready for vector storage."""

    incident: Incident
    embedding: list[float] = Field(..., description="768-dimensional vector from Mistral model")

    # Derived fields for filtering in Weaviate
    embedding_text: str = Field(..., description="Original text that was embedded (for tracking)")
