from __future__ import annotations
from typing import List, Optional, Dict
from uuid import uuid4, UUID
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON
from pgvector.sqlalchemy import Vector


class Restaurant(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    location: Optional[str] = None
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))


class MenuItem(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    restaurant_id: UUID = Field(index=True)
    name: str
    description: str = ""
    ingredients: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    # Localized user-facing content keyed by language code, e.g.
    # {"es": {"description": "...", "ingredients": ["...", ...]}}.
    # English originals stay in `description`/`ingredients` (also used by
    # safety filtering and feature extraction — never overwrite them).
    content_translations: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    allergens: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    dietary_tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    cuisine: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    price: Optional[float] = None
    spice_level: Optional[int] = None
    cooking_method: Optional[str] = None
    course: Optional[str] = None
    features: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    texture: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    richness: Optional[float] = None
    provenance: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    inference_confidence: float = 1.0
    
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(1536)))
    reduced_embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(64)))
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None
    last_embedded_at: Optional[datetime] = None
