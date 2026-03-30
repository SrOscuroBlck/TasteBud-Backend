from __future__ import annotations
from typing import List, Optional, Dict
from datetime import datetime
from uuid import uuid4, UUID
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON


TASTE_AXES = [
    "sweet", "sour", "salty", "bitter", "umami", "fatty", "spicy"
]

TEXTURE_AXES = [
    "crunchy", "creamy", "chewy"
]


class QuestionOption(SQLModel):
    id: str
    label: str
    tags: List[str] = []
    ingredient_keys: List[str] = []


class OnboardingQuestion(SQLModel):
    question_id: UUID
    prompt: str
    options: List[QuestionOption]
    axis_hints: Dict[str, float] = {}


class OnboardingAnswer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question_id: UUID
    chosen_option_id: str
    timestamp: datetime
    user_id: UUID = Field(index=True)


class OnboardingState(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: UUID = Field(index=True)
    active: bool = True
    answered_pairs: List[Dict] = Field(default_factory=list, sa_column=Column(JSON))
    pending_axis_targets: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    confidence: float = 0.0
    current_question_data: Optional[Dict] = Field(default=None, sa_column=Column(JSON))


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    email_verified: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    onboarding_completed: bool = False

    allergies: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    dietary_rules: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    disliked_ingredients: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    liked_ingredients: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    taste_vector: Dict[str, float] = Field(default_factory=lambda: {k: 0.5 for k in TASTE_AXES}, sa_column=Column(JSON))
    taste_uncertainty: Dict[str, float] = Field(default_factory=lambda: {k: 0.5 for k in TASTE_AXES}, sa_column=Column(JSON))
    taste_archetype_id: Optional[UUID] = Field(default=None, index=True)
    cuisine_affinity: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    ingredient_penalties: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    
    permanently_excluded_items: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    preferred_language: str = Field(default="en")

    onboarding_choices: List[Dict] = Field(default_factory=list, sa_column=Column(JSON))
    onboarding_state: Optional[Dict] = Field(default=None, sa_column=Column(JSON))
