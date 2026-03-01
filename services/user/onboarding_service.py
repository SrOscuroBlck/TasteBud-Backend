from __future__ import annotations
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
from sqlmodel import Session, select, select
from config.settings import settings
from models import User, OnboardingState, PopulationStats
from models.user import TASTE_AXES
from models.bayesian_profile import BayesianTasteProfile
from services.features.features import clamp01
from services.features.gpt_helper import generate_onboarding_question
from .archetype_service import get_archetype_by_id, find_closest_archetype
from services.learning.bayesian_profile_service import BayesianProfileService
from utils.logger import setup_logger

logger = setup_logger(__name__)


FALLBACK_QUESTIONS = [
    {
        "prompt": "Would you rather have a cheesy Margherita pizza or a spicy beef taco?",
        "options": [
            {
                "id": "A",
                "label": "Margherita Pizza",
                "tags": ["cheesy", "baked", "umami"],
                "ingredient_keys": ["dough", "tomato", "mozzarella"],
                "axis_impacts": {"umami": 0.3, "fatty": 0.2, "salty": 0.1},
            },
            {
                "id": "B",
                "label": "Spicy Beef Taco",
                "tags": ["spicy", "fried", "umami"],
                "ingredient_keys": ["beef", "chili"],
                "axis_impacts": {"spicy": 0.4, "umami": 0.2, "salty": 0.1},
            },
        ],
    },
    {
        "prompt": "Would you prefer sweet honey-glazed salmon or tangy lemon herb chicken?",
        "options": [
            {
                "id": "A",
                "label": "Honey-Glazed Salmon",
                "tags": ["sweet", "grilled", "tender"],
                "ingredient_keys": ["salmon", "honey", "ginger"],
                "axis_impacts": {"sweet": 0.4, "umami": 0.2, "fatty": 0.1},
            },
            {
                "id": "B",
                "label": "Lemon Herb Chicken",
                "tags": ["tangy", "citrus", "savory"],
                "ingredient_keys": ["chicken", "lemon", "herbs"],
                "axis_impacts": {"sour": 0.4, "bitter": 0.1, "umami": 0.1},
            },
        ],
    },
    {
        "prompt": "Would you choose crispy fried chicken or a fresh garden salad?",
        "options": [
            {
                "id": "A",
                "label": "Crispy Fried Chicken",
                "tags": ["crispy", "fried", "savory"],
                "ingredient_keys": ["chicken", "breading", "oil"],
                "axis_impacts": {"fatty": 0.4, "salty": 0.2, "umami": 0.1},
            },
            {
                "id": "B",
                "label": "Fresh Garden Salad",
                "tags": ["fresh", "crunchy", "light"],
                "ingredient_keys": ["lettuce", "tomato", "cucumber"],
                "axis_impacts": {"bitter": 0.2, "sour": 0.2, "fatty": -0.2},
            },
        ],
    },
    {
        "prompt": "Would you rather have salty french fries or sweet potato fries?",
        "options": [
            {
                "id": "A",
                "label": "Salty French Fries",
                "tags": ["salty", "crispy", "savory"],
                "ingredient_keys": ["potato", "salt", "oil"],
                "axis_impacts": {"salty": 0.4, "fatty": 0.2},
            },
            {
                "id": "B",
                "label": "Sweet Potato Fries",
                "tags": ["sweet", "crispy", "earthy"],
                "ingredient_keys": ["sweet_potato", "oil", "cinnamon"],
                "axis_impacts": {"sweet": 0.4, "fatty": 0.1},
            },
        ],
    },
    {
        "prompt": "Would you prefer rich chocolate lava cake or tangy key lime pie?",
        "options": [
            {
                "id": "A",
                "label": "Chocolate Lava Cake",
                "tags": ["sweet", "rich", "warm"],
                "ingredient_keys": ["chocolate", "butter", "sugar"],
                "axis_impacts": {"sweet": 0.3, "bitter": 0.2, "fatty": 0.2},
            },
            {
                "id": "B",
                "label": "Key Lime Pie",
                "tags": ["tangy", "citrus", "tart"],
                "ingredient_keys": ["lime", "condensed_milk", "graham_cracker"],
                "axis_impacts": {"sour": 0.4, "sweet": 0.2},
            },
        ],
    },
    {
        "prompt": "Would you choose spicy hot wings or mild BBQ ribs?",
        "options": [
            {
                "id": "A",
                "label": "Spicy Hot Wings",
                "tags": ["spicy", "tangy", "crispy"],
                "ingredient_keys": ["chicken", "hot_sauce", "butter"],
                "axis_impacts": {"spicy": 0.5, "salty": 0.2, "fatty": 0.1},
            },
            {
                "id": "B",
                "label": "Mild BBQ Ribs",
                "tags": ["sweet", "smoky", "tender"],
                "ingredient_keys": ["pork", "bbq_sauce", "brown_sugar"],
                "axis_impacts": {"sweet": 0.3, "umami": 0.3, "fatty": 0.1},
            },
        ],
    },
    {
        "prompt": "Would you rather have creamy mac and cheese or light steamed vegetables?",
        "options": [
            {
                "id": "A",
                "label": "Creamy Mac and Cheese",
                "tags": ["creamy", "cheesy", "rich"],
                "ingredient_keys": ["pasta", "cheese", "cream"],
                "axis_impacts": {"fatty": 0.4, "salty": 0.2, "umami": 0.2},
            },
            {
                "id": "B",
                "label": "Steamed Vegetables",
                "tags": ["light", "fresh", "healthy"],
                "ingredient_keys": ["broccoli", "carrot", "zucchini"],
                "axis_impacts": {"bitter": 0.2, "sour": 0.1, "fatty": -0.3},
            },
        ],
    },
]


class OnboardingService:
    def start(self, user: User, session: Session) -> Dict[str, Any]:
        # Deactivate all previous onboarding states
        previous_states = session.exec(
            select(OnboardingState).where(
                OnboardingState.user_id == user.id,
                OnboardingState.active == True
            )
        ).all()
        for prev_state in previous_states:
            prev_state.active = False
            session.add(prev_state)
        session.commit()
        
        # Initialize taste vector from archetype if available
        archetype = None
        if user.taste_archetype_id:
            try:
                archetype = get_archetype_by_id(session, user.taste_archetype_id)
            except Exception:
                archetype = None
        
        # If no archetype assigned, try to find closest match
        if not archetype:
            try:
                user_preferences = {}
                archetype = find_closest_archetype(session, user_preferences)
                user.taste_archetype_id = archetype.id
            except Exception:
                archetype = None
        
        # Initialize taste_vector
        if not user.taste_vector:
            if archetype:
                user.taste_vector = dict(archetype.taste_vector)
            else:
                priors = session.exec(select(PopulationStats)).first()
                if priors and priors.axis_prior_mean:
                    user.taste_vector = dict(priors.axis_prior_mean)
                else:
                    user.taste_vector = {k: 0.5 for k in TASTE_AXES}
        
        # Initialize taste_uncertainty with moderate uncertainty for archetype-based initialization
        if not user.taste_uncertainty:
            if archetype:
                user.taste_uncertainty = {k: 0.3 for k in TASTE_AXES}
            else:
                priors = session.exec(select(PopulationStats)).first()
                if priors and priors.axis_prior_sigma:
                    user.taste_uncertainty = dict(priors.axis_prior_sigma)
                else:
                    user.taste_uncertainty = {k: 0.5 for k in TASTE_AXES}
        
        # Initialize cuisine_affinity
        if not user.cuisine_affinity:
            if archetype and archetype.typical_cuisines:
                user.cuisine_affinity = {cuisine: 0.7 for cuisine in archetype.typical_cuisines}
            else:
                priors = session.exec(select(PopulationStats)).first()
                if priors and priors.cuisine_prior:
                    user.cuisine_affinity = dict(priors.cuisine_prior)
        
        state = OnboardingState(user_id=user.id, active=True, answered_pairs=[], pending_axis_targets=self._top_uncertain_axes(user), confidence=0.0)
        session.add(state)
        session.commit()
        session.refresh(state)
        user.onboarding_state = {"state_id": state.id, "active": True, "confidence": 0.0}
        session.add(user)
        session.commit()
        return self._next_question(session, user, state)

    def answer(self, user: User, question_id: str, chosen_option_id: str, session: Session) -> Dict[str, Any]:
        state = session.exec(select(OnboardingState).where(OnboardingState.user_id == user.id, OnboardingState.active == True)).first()
        if not state:
            return {"complete": True}
        
        last_question_data = state.current_question_data or {}
        options = last_question_data.get("options", [])
        chosen_option = next((opt for opt in options if opt.get("id") == chosen_option_id), None)
        
        if chosen_option:
            axis_impacts = chosen_option.get("axis_impacts", {})
            touched_axes = set()
            updated_taste_vector = dict(user.taste_vector)
            updated_uncertainty = dict(user.taste_uncertainty)
            for axis, delta in axis_impacts.items():
                if axis not in TASTE_AXES:
                    continue
                try:
                    v = float(delta)
                except Exception:
                    continue
                old = updated_taste_vector.get(axis, 0.5)
                updated_taste_vector[axis] = clamp01(old + settings.ONBOARDING_K * v)
                touched_axes.add(axis)
            
            for axis in touched_axes:
                updated_uncertainty[axis] = max(
                    0.0,
                    updated_uncertainty.get(axis, 0.5) - settings.ONBOARDING_SIGMA_STEP,
                )
            
            user.taste_vector = updated_taste_vector
            user.taste_uncertainty = updated_uncertainty
        
        choice_record = {
            "question_id": question_id,
            "prompt": last_question_data.get("prompt", ""),
            "chosen": chosen_option_id,
            "chosen_label": chosen_option.get("label", "") if chosen_option else "",
            "chosen_tags": chosen_option.get("tags", []) if chosen_option else [],
            "chosen_ingredients": chosen_option.get("ingredient_keys", []) if chosen_option else [],
            "axis_impacts_applied": chosen_option.get("axis_impacts", {}) if chosen_option else {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        state.answered_pairs = state.answered_pairs + [choice_record]
        
        if not user.onboarding_choices:
            user.onboarding_choices = []
        user.onboarding_choices = user.onboarding_choices + [choice_record]
        
        avg_sigma = sum(user.taste_uncertainty.values()) / max(1, len(user.taste_uncertainty))
        state.confidence = 1.0 - avg_sigma
        user.last_updated = datetime.utcnow()
        session.add(user)
        session.add(state)
        session.commit()
        session.refresh(state)
        
        if state.confidence >= settings.ONBOARDING_EARLY_STOP_CONFIDENCE or len(state.answered_pairs) >= settings.ONBOARDING_MAX_QUESTIONS:
            state.active = False
            session.add(state)
            session.commit()
            
            self._calculate_cuisine_affinity_from_choices(user, state, session)
            self._ensure_bayesian_profile(user, session)
            
            return {"complete": True}
        return self._next_question(session, user, state)

    def _next_question(self, session: Session, user: User, state: OnboardingState) -> Dict[str, Any]:
        target_axes = self._top_uncertain_axes(user)
        previous_prompts = [pair.get("prompt", "") for pair in state.answered_pairs if pair.get("prompt")]
        q = generate_onboarding_question(target_axes, user.allergies or [], previous_prompts)
        if not q:
            fallback_index = len(state.answered_pairs) % len(FALLBACK_QUESTIONS)
            qf = FALLBACK_QUESTIONS[fallback_index]
            q = {"question_id": str(uuid4()), **qf}
            logger.warning(
                "Using fallback onboarding question (GPT failed or unavailable)",
                extra={
                    "user_id": str(user.id),
                    "fallback_index": fallback_index,
                    "target_axes": target_axes,
                    "answered_count": len(state.answered_pairs),
                },
            )
        
        state.current_question_data = {
            "question_id": q["question_id"],
            "prompt": q.get("prompt", ""),
            "options": q.get("options", []),
        }
        session.add(state)
        session.commit()
        return q

    def _top_uncertain_axes(self, user: User) -> List[str]:
        items = sorted(user.taste_uncertainty.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in items[:3]]
    
    def _calculate_cuisine_affinity_from_choices(self, user: User, state: OnboardingState, session: Session) -> None:
        """Calculate cuisine affinity based on ingredients chosen during onboarding"""
        # Mapping of ingredients to cuisines
        ingredient_cuisine_map = {
            # Italian
            "dough": ["Italian"], "mozzarella": ["Italian"], "pasta": ["Italian"], 
            "parmesan": ["Italian"], "basil": ["Italian"], "olive_oil": ["Italian"],
            
            # Mexican
            "chili": ["Mexican"], "tortilla": ["Mexican"], "jalapeño": ["Mexican"],
            "cilantro": ["Mexican"], "lime": ["Mexican"], "cumin": ["Mexican"],
            "avocado": ["Mexican"],
            
            # Asian (Japanese, Chinese, Thai)
            "soy_sauce": ["Japanese", "Chinese"], "ginger": ["Asian"], 
            "sesame": ["Asian"], "rice": ["Asian"], "noodles": ["Asian"],
            "tofu": ["Asian"], "miso": ["Japanese"], "wasabi": ["Japanese"],
            
            # Indian
            "curry": ["Indian"], "turmeric": ["Indian"], "cardamom": ["Indian"],
            "coriander": ["Indian"], "ghee": ["Indian"],
            
            # American
            "beef": ["American"], "bacon": ["American"], "cheddar": ["American"],
            
            # Mediterranean
            "lamb": ["Mediterranean"], "feta": ["Mediterranean"], 
            "olive": ["Mediterranean"], "lemon": ["Mediterranean"],
            
            # General proteins/vegetables (multiple cuisines)
            "chicken": ["American", "Asian", "Mexican", "Mediterranean"],
            "tomato": ["Italian", "Mexican", "Mediterranean"],
            "garlic": ["Italian", "Asian", "Mexican", "Mediterranean"],
        }
        
        # Count cuisine occurrences from chosen ingredients
        cuisine_counts = {}
        total_choices = 0
        
        for answer in state.answered_pairs:
            ingredients = answer.get("ingredients", [])
            for ingredient in ingredients:
                cuisines = ingredient_cuisine_map.get(ingredient, [])
                for cuisine in cuisines:
                    cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
                    total_choices += 1
        
        # If no cuisines detected from ingredients, use population priors
        if total_choices == 0:
            priors = session.exec(select(PopulationStats)).first()
            if priors and priors.cuisine_prior:
                user.cuisine_affinity = dict(priors.cuisine_prior)
            else:
                user.cuisine_affinity = {}
            session.add(user)
            session.commit()
            return
        
        # Calculate affinity scores (normalized by total choices)
        cuisine_affinity = {}
        for cuisine, count in cuisine_counts.items():
            # Score between 0.3 (min) and 1.0 (max) based on frequency
            score = 0.3 + (0.7 * count / max(total_choices, 1))
            cuisine_affinity[cuisine] = min(1.0, score)
        
        # Add common cuisines not seen with low default scores
        all_cuisines = ["Italian", "Mexican", "Japanese", "Chinese", "Indian", "American", "Mediterranean", "Thai"]
        for cuisine in all_cuisines:
            if cuisine not in cuisine_affinity:
                cuisine_affinity[cuisine] = 0.2  # Low baseline for unseen cuisines
        
        user.cuisine_affinity = cuisine_affinity
        session.add(user)
        session.commit()
    
    def _ensure_bayesian_profile(self, user: User, session: Session) -> None:
        existing_profile = session.exec(
            select(BayesianTasteProfile).where(BayesianTasteProfile.user_id == user.id)
        ).first()
        
        if existing_profile:
            return
        
        bayesian_service = BayesianProfileService()
        profile = bayesian_service.create_profile_from_user(session, user)
        session.add(profile)
        session.commit()
