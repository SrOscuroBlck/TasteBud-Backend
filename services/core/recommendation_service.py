from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from sqlmodel import Session, select
from models import User, MenuItem, PopulationStats, RecommendationSession, RecommendationFeedback, UserOrderHistory, BayesianTasteProfile
from models.query import ParsedQuery
from services.features.features import cosine_similarity, has_allergen, violates_diet
from services.features.gpt_helper import generate_rationale
from services.core.retrieval_service import RetrievalService
from services.core.reranking_service import RerankingService, RecommendationContext
from services.ml.ml_reranking_service import MLRerankingService
from services.explanation.explanation_service import ExplanationService
from services.context.context_enhancement_service import ContextEnhancementService
from services.learning.in_session_learning_service import InSessionLearningService
from services.composition.meal_composition_service import MealCompositionService
from services.explanation.explanation_enhancement_service import ExplanationEnhancementService
from services.user.interaction_history_service import InteractionHistoryService
from services.evaluation.confidence_service import ConfidenceService
from services.composition.query_service import QueryParsingService
from services.diversity.mmr_service import MMRService, DiversityConstraints
from services.diversity.cross_encoder_service import CrossEncoderService
from services.core.session_service import RecommendationSessionService
from services.learning.bayesian_profile_service import BayesianProfileService
from config.settings import settings
import math
from datetime import datetime, timedelta
from utils.logger import setup_logger

logger = setup_logger(__name__)


def time_decay_score(ts: Optional[datetime], half_life_days: int) -> float:
    if not ts:
        return 1.0
    days = (datetime.utcnow() - ts).days
    return 0.5 ** (days / max(1, half_life_days))


class RecommendationService:
    def __init__(self, use_new_pipeline: bool = True, use_ml_reranking: bool = True):
        self.use_new_pipeline = use_new_pipeline
        self.use_ml_reranking = use_ml_reranking
        self.retrieval_service = RetrievalService() if use_new_pipeline else None
        self.reranking_service = None
        self.ml_reranking_service = None
        self.explanation_service = ExplanationService() if use_new_pipeline else None
        self.context_service = ContextEnhancementService()
        self.in_session_learning = InSessionLearningService()
        self.meal_composition = MealCompositionService()
        self.explanation_enhancement = ExplanationEnhancementService()
        self.interaction_history_service = InteractionHistoryService()
        self.confidence_service = ConfidenceService()
        self.query_parsing_service = QueryParsingService()
        self.mmr_service = MMRService()
        self.cross_encoder_service = CrossEncoderService()
        self.bayesian_profile_service = BayesianProfileService()
    
    def recommend(
        self,
        session: Session,
        user: User,
        restaurant_id: Optional[str] = None,
        top_n: int = 10,
        budget: Optional[float] = None,
        time_of_day: Optional[str] = None,
        mood: Optional[str] = None,
        occasion: Optional[str] = None,
        course_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.use_new_pipeline:
            return self._recommend_new_pipeline(
                session, user, restaurant_id, top_n,
                budget, time_of_day, mood, occasion, course_preference
            )
        else:
            return self._recommend_legacy(
                session, user, restaurant_id, top_n,
                budget, time_of_day
            )
    
    def _recommend_new_pipeline(
        self,
        session: Session,
        user: User,
        restaurant_id: Optional[str],
        top_n: int,
        budget: Optional[float],
        time_of_day: Optional[str],
        mood: Optional[str],
        occasion: Optional[str],
        course_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(
            "Generating recommendations with new pipeline",
            extra={
                "user_id": str(user.id),
                "restaurant_id": restaurant_id,
                "top_n": top_n,
                "course_preference": course_preference
            }
        )
        
        try:
            candidates = self.retrieval_service.retrieve_candidates(
                session=session,
                user=user,
                k=max(top_n * 3, 30),
                restaurant_id=restaurant_id,
                budget=budget,
                course_filter=course_preference,
                time_of_day=time_of_day
            )
        except Exception as e:
            logger.error(
                "Retrieval failed, falling back to legacy",
                extra={"error": str(e)},
                exc_info=True
            )
            return self._recommend_legacy(
                session, user, restaurant_id, top_n, budget, time_of_day
            )
        
        if not candidates:
            return {"items": [], "warnings": ["no_safe_items"]}
        
        pop_stats = session.exec(select(PopulationStats)).first()
        
        # Use ML reranking if enabled and model available
        if self.use_ml_reranking:
            if not self.ml_reranking_service:
                self.ml_reranking_service = MLRerankingService(population_stats=pop_stats)
            
            context = RecommendationContext(
                time_of_day=time_of_day,
                budget=budget,
                mood=mood,
                occasion=occasion,
                course_preference=course_preference
            )
            
            logger.info("Using ML reranking service")
            ranked_items = self.ml_reranking_service.rerank(
                candidates=candidates,
                user=user,
                context=context,
                session=session,
                top_n=top_n
            )
        else:
            # Fall back to rule-based reranking
            if not self.reranking_service:
                self.reranking_service = RerankingService(population_stats=pop_stats)
            
            context = RecommendationContext(
                time_of_day=time_of_day,
                budget=budget,
                mood=mood,
                occasion=occasion,
                course_preference=course_preference
            )
            
            logger.info("Using rule-based reranking service")
            ranked_items = self.reranking_service.rerank(
                candidates=candidates,
                user=user,
                context=context,
                top_n=top_n
            )
        
        context_dict = {
            "time_of_day": time_of_day,
            "budget": budget,
            "mood": mood,
            "occasion": occasion
        }
        explanations = self.explanation_service.generate_explanations(
            ranked_items=ranked_items,
            user=user,
            context=context_dict
        )
        
        results = []
        for ranked_item, explanation in zip(ranked_items, explanations):
            item = ranked_item.item
            user_sorted = sorted(
                user.taste_vector.items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            matched = [
                k for k, v in user_sorted
                if item.features.get(k, 0.0) > 0.5
            ][:3]
            
            results.append({
                "item_id": str(item.id),
                "name": item.name,
                "score": round(ranked_item.final_score, 3),
                "matched_axes": matched,
                "reason": explanation,
                "safety_flags": [],
                "cuisine": item.cuisine,
                "price": item.price,
                "image_url": item.provenance.get("image_url") if item.provenance else None,
                "confidence": ranked_item.confidence,
                "provenance": {
                    "source": item.provenance.get("source", "ingested"),
                    "inference_confidence": item.inference_confidence
                },
                "ranking_factors": {
                    k: round(v, 3)
                    for k, v in ranked_item.ranking_factors.items()
                }
            })
        
        logger.info(
            "Recommendations generated successfully",
            extra={"user_id": str(user.id), "result_count": len(results)}
        )
        
        return {"items": results}
    
    def recommend_from_query(
        self,
        session: Session,
        user: User,
        query: str,
        top_n: int = 10,
        budget: Optional[float] = None,
        diversity_weight: float = 0.3,
        use_cross_encoder: bool = False,
        diversity_constraints: Optional[DiversityConstraints] = None
    ) -> Dict[str, Any]:
        if not query:
            raise ValueError("query cannot be empty")
        
        if not self.use_new_pipeline:
            raise ValueError("query-based recommendations require new pipeline (use_new_pipeline=True)")
        
        logger.info(
            "Generating query-based recommendations",
            extra={
                "user_id": str(user.id),
                "query": query,
                "top_n": top_n,
                "diversity_weight": diversity_weight,
                "use_cross_encoder": use_cross_encoder
            }
        )
        
        try:
            parsed_query = self.query_parsing_service.parse_query(query)
            
            logger.info(
                "Query parsed",
                extra={
                    "intent": parsed_query.intent.value,
                    "modifiers": [m.value for m in parsed_query.modifiers]
                }
            )
            
            candidates = self.retrieval_service.retrieve_candidates_from_query(
                session=session,
                user=user,
                parsed_query=parsed_query,
                k=max(top_n * 3, 50),
                budget=budget
            )
            
            if not candidates:
                return {
                    "items": [],
                    "warnings": ["no_candidates_found"],
                    "query_info": {
                        "raw_query": query,
                        "intent": parsed_query.intent.value,
                        "modifiers": [m.value for m in parsed_query.modifiers]
                    }
                }
            
            if use_cross_encoder and self.cross_encoder_service.is_available:
                logger.info("Applying cross-encoder reranking")
                scored_candidates = self.cross_encoder_service.rerank_parsed_query_results(
                    parsed_query=parsed_query,
                    candidates=candidates,
                    top_k=min(len(candidates), top_n * 2)
                )
                candidates = [item for item, score in scored_candidates]
            
            if diversity_weight > 0:
                logger.info("Applying MMR diversity reranking")
                final_items = self.mmr_service.rerank_with_mmr(
                    candidates=candidates,
                    user_taste_vector=user.taste_vector,
                    k=top_n,
                    diversity_weight=diversity_weight,
                    constraints=diversity_constraints
                )
            else:
                final_items = candidates[:top_n]
            
            results = []
            for item in final_items:
                item_data = {
                    "item_id": str(item.id),
                    "name": item.name,
                    "description": item.description,
                    "course": item.course,
                    "price": item.price,
                    "image_url": item.provenance.get("image_url") if item.provenance else None,
                    "cuisine": item.cuisine,
                    "dietary_tags": item.dietary_tags,
                    "features": item.features,
                    "explanation": f"Matches your query: {query}"
                }
                results.append(item_data)
            
            logger.info(
                "Query-based recommendations generated successfully",
                extra={
                    "user_id": str(user.id),
                    "query": query,
                    "result_count": len(results)
                }
            )
            
            return {
                "items": results,
                "query_info": {
                    "raw_query": query,
                    "intent": parsed_query.intent.value,
                    "modifiers": [m.value for m in parsed_query.modifiers],
                    "cuisine_filter": parsed_query.cuisine_filter,
                    "taste_adjustments": parsed_query.taste_adjustments
                },
                "diversity_score": self.mmr_service._compute_diversity_score(final_items) if diversity_weight > 0 else None
            }
            
        except Exception as e:
            logger.error(
                "Query-based recommendation failed",
                extra={"error": str(e), "query": query},
                exc_info=True
            )
            raise
    
    def _recommend_legacy(
        self,
        session: Session,
        user: User,
        restaurant_id: Optional[str],
        top_n: int,
        budget: Optional[float],
        time_of_day: Optional[str]
    ) -> Dict[str, Any]:
        logger.info("Using legacy recommendation pipeline")
        
        q = select(MenuItem)
        if restaurant_id:
            from uuid import UUID
            q = q.where(MenuItem.restaurant_id == UUID(restaurant_id))
        items: List[MenuItem] = session.exec(q).all()

        # 2) hard filters
        safe: List[MenuItem] = []
        user_all = set(map(str.lower, user.allergies))
        for it in items:
            # explicit allergens list check
            if user_all.intersection(set(map(str.lower, it.allergens))):
                continue
            # deterministic ingredient mapping check
            if has_allergen(user.allergies, it.ingredients, explicit_allergens=it.allergens):
                continue
            if violates_diet(user.dietary_rules, it.dietary_tags):
                continue
            if budget is not None and it.price is not None and it.price > budget:
                continue
            safe.append(it)

        if not safe:
            return {"items": [], "warnings": ["no_safe_items"]}

        # 3) scoring
        pop = session.exec(select(PopulationStats)).first()
        pop_global = pop.item_popularity_global if pop else {}
        pop_rest = pop.item_popularity_by_restaurant if pop else {}

        def cuisine_aff(it: MenuItem) -> float:
            return max((user.cuisine_affinity.get(c, 0.0) for c in it.cuisine), default=0.0)

        def popularity(it: MenuItem) -> float:
            base = pop_global.get(str(it.id), 0.0) + pop_rest.get(str(it.restaurant_id), 0.0)
            return min(1.0, base)

        base_scores: Dict[str, float] = {}
        for it in safe:
            s = cosine_similarity(user.taste_vector, it.features)
            s += settings.LAMBDA_CUISINE * cuisine_aff(it)
            s += settings.LAMBDA_POP * popularity(it)
            # liked/disliked penalties
            if any(ing in set(map(str.lower, user.disliked_ingredients)) for ing in map(str.lower, it.ingredients)):
                s -= 0.1
            if any(ing in set(map(str.lower, user.liked_ingredients)) for ing in map(str.lower, it.ingredients)):
                s += 0.05
            # provenance discount
            if it.provenance.get("source") == "gpt_inferred":
                conf = it.inference_confidence or 0.5
                s *= (1.0 - settings.GPT_CONFIDENCE_DISCOUNT * (1.0 - conf))
            base_scores[str(it.id)] = max(0.0, min(1.0, s))

        # 5) diversification (MMR)
        selected: List[MenuItem] = []
        selected_ids: set = set()
        alpha = settings.MMR_ALPHA
        feats = {str(it.id): it.features for it in safe}
        while len(selected) < min(top_n, len(safe)):
            best_item = None
            best_score = -math.inf
            for it in safe:
                if str(it.id) in selected_ids:
                    continue
                diversity_penalty = 0.0
                if selected:
                    max_sim = max(cosine_similarity(feats[str(it.id)], feats[str(s.id)]) for s in selected)
                    diversity_penalty = (1 - alpha) * max_sim
                cand = alpha * base_scores[str(it.id)] - diversity_penalty
                if cand > best_score:
                    best_score = cand
                    best_item = it
            if not best_item:
                break
            selected.append(best_item)
            selected_ids.add(str(best_item.id))

        # 6) explainability
        results = []
        for it in selected:
            # matched axes: top positive contribution axes from item where user prefers high
            user_sorted = sorted(user.taste_vector.items(), key=lambda kv: kv[1], reverse=True)
            matched = [k for k, v in user_sorted if it.features.get(k, 0.0) > 0.5][:3]
            reason = generate_rationale({
                "user_axes": {k: user.taste_vector[k] for k in matched},
                "ingredients": it.ingredients,
                "cuisine": it.cuisine,
            }) or "Matched your tastes and restrictions."
            results.append({
                "item_id": str(it.id),
                "name": it.name,
                "score": round(base_scores[str(it.id)], 3),
                "matched_axes": matched,
                "reason": reason,
                "safety_flags": [],
                "cuisine": it.cuisine,
                "price": it.price,
                "image_url": it.provenance.get("image_url") if it.provenance else None,
                "confidence": it.inference_confidence,
                "provenance": {"source": it.provenance.get("source", "ingested"), "inference_confidence": it.inference_confidence},
            })

        return {"items": results}
    
    def recommend_with_session(
        self,
        session: Session,
        user: User,
        recommendation_session: RecommendationSession,
        top_n: int = 10,
        iteration: int = 1
    ) -> Dict[str, Any]:
        logger.info(
            "Session-based recommendation starting",
            extra={
                "session_id": str(recommendation_session.id),
                "user_id": str(user.id),
                "meal_intent": recommendation_session.meal_intent,
                "iteration": iteration
            }
        )
        
        from uuid import UUID as UUIDType
        restaurant_id_str = str(recommendation_session.restaurant_id)
        
        q = select(MenuItem).where(MenuItem.restaurant_id == UUIDType(restaurant_id_str))
        all_items: List[MenuItem] = session.exec(q).all()
        
        breakfast_courses = ["breakfast", "brunch"]
        restaurant_has_breakfast = any(
            (item.course or "").lower() in breakfast_courses
            for item in all_items
        )
        
        safe: List[MenuItem] = []
        user_all = set(map(str.lower, user.allergies))
        
        filtered_counts = {
            "allergen": 0,
            "diet": 0,
            "budget": 0,
            "session_excluded": 0
        }
        
        logger.info(
            "Starting safety filtering",
            extra={
                "user_id": str(user.id),
                "session_id": str(recommendation_session.id),
                "total_items": len(all_items),
                "user_allergies": list(user.allergies),
                "user_dietary_rules": list(user.dietary_rules),
                "session_excluded_count": len(recommendation_session.excluded_items)
            }
        )
        
        for it in all_items:
            if user_all.intersection(set(map(str.lower, it.allergens))):
                filtered_counts["allergen"] += 1
                continue
            if has_allergen(user.allergies, it.ingredients, explicit_allergens=it.allergens):
                filtered_counts["allergen"] += 1
                continue
            if violates_diet(user.dietary_rules, it.dietary_tags):
                filtered_counts["diet"] += 1
                continue
            if recommendation_session.budget and it.price and it.price > recommendation_session.budget * 1.2:
                filtered_counts["budget"] += 1
                continue
            
            item_id_str = str(it.id)
            if item_id_str in recommendation_session.excluded_items:
                filtered_counts["session_excluded"] += 1
                continue
            
            safe.append(it)
        
        logger.info(
            "Safety filtering completed",
            extra={
                "user_id": str(user.id),
                "session_id": str(recommendation_session.id),
                "initial_count": len(all_items),
                "safe_count": len(safe),
                "filtered_by_allergen": filtered_counts["allergen"],
                "filtered_by_diet": filtered_counts["diet"],
                "filtered_by_budget": filtered_counts["budget"],
                "filtered_by_session_exclusion": filtered_counts["session_excluded"]
            }
        )
        
        if not safe:
            logger.warning(
                "No safe items after filtering",
                extra={
                    "user_id": str(user.id),
                    "session_id": str(recommendation_session.id),
                    "filtered_counts": filtered_counts
                }
            )
            return {"items": [], "warnings": ["no_safe_items"]}
        
        skip_time_filter_intents = [
            "dessert_only",
            "beverage_only",
            "appetizer_only",
            "light_snack",
        ]
        apply_strict_time_filter = recommendation_session.meal_intent not in skip_time_filter_intents
        
        time_filtered = self.context_service.apply_hard_time_filters(
            safe,
            recommendation_session.detected_hour,
            strict=apply_strict_time_filter
        )
        
        logger.info(
            "Time filtering completed",
            extra={
                "user_id": str(user.id),
                "session_id": str(recommendation_session.id),
                "before_count": len(safe),
                "after_count": len(time_filtered),
                "filtered_count": len(safe) - len(time_filtered),
                "time_of_day": recommendation_session.time_of_day,
                "detected_hour": recommendation_session.detected_hour,
                "strict_filtering": apply_strict_time_filter
            }
        )
        
        intent_filtered = self.context_service.apply_meal_intent_filters(
            time_filtered,
            recommendation_session.meal_intent,
            recommendation_session.hunger_level
        )
        
        logger.info(
            "Intent filtering completed",
            extra={
                "user_id": str(user.id),
                "session_id": str(recommendation_session.id),
                "before_count": len(time_filtered),
                "after_count": len(intent_filtered),
                "filtered_count": len(time_filtered) - len(intent_filtered),
                "meal_intent": recommendation_session.meal_intent,
                "hunger_level": recommendation_session.hunger_level
            }
        )
        
        if not intent_filtered:
            logger.warning(
                "No items match meal intent after filtering",
                extra={
                    "session_id": str(recommendation_session.id),
                    "meal_intent": recommendation_session.meal_intent,
                    "time_filtered_count": len(time_filtered),
                    "safe_count": len(safe),
                }
            )
            return {"items": [], "warnings": ["no_items_for_intent"]}
        
        order_history = session.exec(
            select(UserOrderHistory).where(UserOrderHistory.user_id == user.id)
        ).all()
        
        user_interaction_history = self.interaction_history_service.get_all_user_history(
            db_session=session,
            user_id=user.id,
        )

        candidates = self._apply_cross_session_filters(
            db_session=session,
            items=intent_filtered,
            user=user,
            interaction_history=user_interaction_history,
            order_history=order_history,
        )

        from services.ml.llm_reranking_service import (
            rerank_single_items,
            compose_full_meal,
            compose_partial_meal_llm,
            _matches_course_slot,
        )

        session_feedback = session.exec(
            select(RecommendationFeedback).where(
                RecommendationFeedback.session_id == recommendation_session.id
            )
        ).all()

        items_map = {str(item.id): item for item in intent_filtered}

        session_items_shown = recommendation_session.items_shown or []

        if recommendation_session.meal_intent == "full_meal":
            # For full_meal: separate by course, rank each independently, then combine
            # This ensures each course type gets its best candidates to the LLM
            appetizers = [c for c in candidates if _matches_course_slot(c.course, "appetizer")]
            mains = [c for c in candidates if _matches_course_slot(c.course, "main")]
            desserts = [c for c in candidates if _matches_course_slot(c.course, "dessert")]

            # Fresh-first + taste ranking per course
            per_course = max(5, top_n * 2 // 3)
            appetizers = self._rank_by_taste_affinity(
                candidates=self._prioritize_fresh_candidates(
                    appetizers, user_interaction_history, min_candidates=per_course
                ),
                user=user,
            )[:per_course]
            mains = self._rank_by_taste_affinity(
                candidates=self._prioritize_fresh_candidates(
                    mains, user_interaction_history, min_candidates=per_course
                ),
                user=user,
            )[:per_course]
            desserts = self._rank_by_taste_affinity(
                candidates=self._prioritize_fresh_candidates(
                    desserts, user_interaction_history, min_candidates=per_course
                ),
                user=user,
            )[:per_course]

            candidates = appetizers + mains + desserts

            logger.info(
                "Course-balanced candidates for full_meal",
                extra={
                    "session_id": str(recommendation_session.id),
                    "appetizers": len(appetizers),
                    "mains": len(mains),
                    "desserts": len(desserts),
                    "total": len(candidates),
                },
            )

            candidate_id_set = {str(item.id) for item in candidates}

            pending = recommendation_session.pending_partial_regen
            if pending and pending.get("accepted"):
                compositions_raw = compose_partial_meal_llm(
                    candidates=candidates,
                    user=user,
                    rec_session=recommendation_session,
                    feedback_list=session_feedback,
                    items_map=items_map,
                    accepted=pending["accepted"],
                    rejected=pending["rejected"],
                    num_compositions=max(1, top_n // 3),
                    interaction_history=user_interaction_history,
                    items_shown=session_items_shown,
                    order_history=order_history,
                )
                recommendation_session.pending_partial_regen = None
            else:
                compositions_raw = compose_full_meal(
                    candidates=candidates,
                    user=user,
                    rec_session=recommendation_session,
                    feedback_list=session_feedback,
                    items_map=items_map,
                    num_compositions=max(1, top_n // 3),
                    interaction_history=user_interaction_history,
                    items_shown=session_items_shown,
                    order_history=order_history,
                )

            results = []
            session_service = RecommendationSessionService()
            base_iteration = recommendation_session.iteration_count

            for idx, comp in enumerate(compositions_raw):
                from uuid import UUID as UUIDType

                appetizer = session.get(MenuItem, UUIDType(comp["appetizer_id"]))
                main_item = session.get(MenuItem, UUIDType(comp["main_id"]))
                dessert = session.get(MenuItem, UUIDType(comp["dessert_id"]))

                if not all([appetizer, main_item, dessert]):
                    continue

                if idx == 0:
                    session_service.set_active_composition(
                        db_session=session,
                        session_id=recommendation_session.id,
                        composition_id=f"llm-{base_iteration}-{idx}",
                        appetizer_id=appetizer.id,
                        main_id=main_item.id,
                        dessert_id=dessert.id,
                    )

                for shown_item in [appetizer, main_item, dessert]:
                    try:
                        self.interaction_history_service.record_item_shown(
                            db_session=session,
                            user_id=user.id,
                            item_id=shown_item.id,
                            session_id=recommendation_session.id,
                        )
                    except Exception:
                        pass

                session_service.add_items_shown(
                    db_session=session,
                    session_id=recommendation_session.id,
                    item_ids=[appetizer.id, main_item.id, dessert.id],
                )

                base_scores = {
                    str(appetizer.id): 0.9,
                    str(main_item.id): 0.9,
                    str(dessert.id): 0.9,
                }

                total_price = sum(
                    i.price for i in [appetizer, main_item, dessert] if i.price
                )

                results.append({
                    "composition_id": f"llm-{base_iteration}-{idx}",
                    "items": [
                        self._format_item_response(
                            session, appetizer, user, recommendation_session,
                            base_scores, order_history, user_interaction_history,
                            restaurant_has_breakfast,
                        ),
                        self._format_item_response(
                            session, main_item, user, recommendation_session,
                            base_scores, order_history, user_interaction_history,
                            restaurant_has_breakfast,
                        ),
                        self._format_item_response(
                            session, dessert, user, recommendation_session,
                            base_scores, order_history, user_interaction_history,
                            restaurant_has_breakfast,
                        ),
                    ],
                    "total_price": total_price,
                    "explanation": comp.get("meal_reasoning", ""),
                })

            if results:
                return {"items": results, "type": "composition"}

            return {"items": [], "warnings": ["no_valid_compositions"]}

        # Non-full_meal intents: rank all candidates together
        candidates = self._prioritize_fresh_candidates(
            candidates=candidates,
            interaction_history=user_interaction_history,
            min_candidates=top_n * 2,
        )
        candidates = self._rank_by_taste_affinity(
            candidates=candidates,
            user=user,
        )
        candidate_id_set = {str(item.id) for item in candidates}

        llm_ranked = rerank_single_items(
            candidates=candidates,
            user=user,
            rec_session=recommendation_session,
            feedback_list=session_feedback,
            items_map=items_map,
            top_n=top_n,
            interaction_history=user_interaction_history,
            items_shown=session_items_shown,
            order_history=order_history,
        )

        top_items: list[MenuItem] = []
        for entry in llm_ranked:
            if entry["item_id"] not in candidate_id_set:
                continue
            item = items_map.get(entry["item_id"])
            if item:
                top_items.append(item)

        top_items = self._apply_dislike_penalty_reorder(
            top_items, user_interaction_history
        )

        base_scores = {}
        for rank_idx, item in enumerate(top_items):
            score = max(0.01, 1.0 - rank_idx * (0.9 / max(len(top_items), 1)))
            base_scores[str(item.id)] = round(score, 3)

        logger.info(
            "LLM reranking applied to session pipeline",
            extra={
                "session_id": str(recommendation_session.id),
                "user_id": str(user.id),
                "item_count": len(top_items),
                "iteration": recommendation_session.iteration_count,
            },
        )

        for item in top_items:
            try:
                self.interaction_history_service.record_item_shown(
                    db_session=session,
                    user_id=user.id,
                    item_id=item.id,
                    session_id=recommendation_session.id,
                )
            except Exception:
                pass

        session_service = RecommendationSessionService()
        session_service.add_items_shown(
            db_session=session,
            session_id=recommendation_session.id,
            item_ids=[item.id for item in top_items],
        )

        results = []
        for it in top_items:
            formatted = self._format_item_response(
                session, it, user, recommendation_session, base_scores,
                order_history, user_interaction_history, restaurant_has_breakfast,
            )
            llm_entry = next(
                (e for e in llm_ranked if e["item_id"] == str(it.id)), {}
            )
            if llm_entry.get("reason"):
                formatted["llm_reason"] = llm_entry["reason"]
            results.append(formatted)

        return {"items": results, "type": "single"}
    
    def _apply_dislike_penalty_reorder(
        self,
        items: list[MenuItem],
        interaction_history: Dict,
    ) -> list[MenuItem]:
        DISLIKE_RANK_PENALTY = 6
        INGREDIENT_PENALTY_THRESHOLD = 0.3

        non_penalized: list[MenuItem] = []
        penalized: list[MenuItem] = []

        for item in items:
            item_history = interaction_history.get(item.id)
            if item_history and (item_history.was_disliked or item_history.was_dismissed):
                penalized.append(item)
            else:
                non_penalized.append(item)

        if not penalized:
            return items

        reordered = list(non_penalized)
        insert_position = min(len(reordered), DISLIKE_RANK_PENALTY)
        for p_item in penalized:
            reordered.insert(insert_position, p_item)
            insert_position += 1

        logger.info(
            "Applied dislike/dismiss penalty reorder",
            extra={
                "penalized_count": len(penalized),
                "penalized_names": [p.name for p in penalized],
                "pushed_to_position": DISLIKE_RANK_PENALTY,
                "total_items": len(reordered),
            },
        )

        return reordered

    def _rank_by_taste_affinity(
        self,
        candidates: list[MenuItem],
        user: User,
    ) -> list[MenuItem]:
        TASTE_WEIGHT = 0.55
        CUISINE_WEIGHT = 0.25
        INGREDIENT_WEIGHT = 0.20

        user_taste = user.taste_vector or {}
        user_cuisine = user.cuisine_affinity or {}
        liked_ingredients = set(
            ing.lower() for ing in (user.liked_ingredients or [])
        )
        disliked_ingredients = set(
            ing.lower() for ing in (user.disliked_ingredients or [])
        )

        has_taste = bool(user_taste)
        has_cuisine = bool(user_cuisine)
        has_ingredient_prefs = bool(liked_ingredients or disliked_ingredients)

        if not has_taste and not has_cuisine and not has_ingredient_prefs:
            return candidates

        scored: list[tuple[MenuItem, float]] = []

        for item in candidates:
            taste_score = 0.0
            if has_taste and item.features:
                taste_score = cosine_similarity(user_taste, item.features)

            cuisine_score = 0.0
            if has_cuisine and item.cuisine:
                item_cuisines = [c.lower() for c in item.cuisine]
                matching_affinities = [
                    affinity
                    for cuisine_name, affinity in user_cuisine.items()
                    if cuisine_name.lower() in item_cuisines
                ]
                if matching_affinities:
                    cuisine_score = max(matching_affinities)

            ingredient_score = 0.5
            if has_ingredient_prefs and item.ingredients:
                item_ings = set(ing.lower() for ing in item.ingredients)
                liked_overlap = len(item_ings & liked_ingredients)
                disliked_overlap = len(item_ings & disliked_ingredients)
                total_ings = max(len(item_ings), 1)
                ingredient_score = 0.5 + (liked_overlap - disliked_overlap * 2) / total_ings
                ingredient_score = max(0.0, min(1.0, ingredient_score))

            combined = (
                TASTE_WEIGHT * taste_score
                + CUISINE_WEIGHT * cuisine_score
                + INGREDIENT_WEIGHT * ingredient_score
            )
            scored.append((item, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "Taste-affinity ranking applied",
            extra={
                "candidate_count": len(candidates),
                "has_taste_vector": has_taste,
                "has_cuisine_affinity": has_cuisine,
                "has_ingredient_prefs": has_ingredient_prefs,
                "top_3_scores": [
                    {"name": item.name, "score": round(score, 3)}
                    for item, score in scored[:3]
                ],
                "bottom_3_scores": [
                    {"name": item.name, "score": round(score, 3)}
                    for item, score in scored[-3:]
                ],
            },
        )

        return [item for item, _ in scored]

    def _apply_cross_session_filters(
        self,
        db_session: Session,
        items: list[MenuItem],
        user: User,
        interaction_history: Dict,
        order_history: list[UserOrderHistory],
    ) -> list[MenuItem]:
        DISLIKE_HARD_EXCLUDE_DAYS = 90
        DISLIKE_DECAY_HALF_LIFE_DAYS = 120
        ORDER_PENALTY_DAYS = 30

        hard_cutoff = datetime.utcnow() - timedelta(days=DISLIKE_HARD_EXCLUDE_DAYS)

        dislike_rows = db_session.exec(
            select(RecommendationFeedback.item_id, RecommendationFeedback.timestamp)
            .where(RecommendationFeedback.session_id.in_(
                select(RecommendationSession.id)
                .where(RecommendationSession.user_id == user.id)
            ))
            .where(RecommendationFeedback.feedback_type == "dislike")
        ).all()

        dislike_timestamps: dict[UUID, datetime] = {}
        for item_id, ts in dislike_rows:
            existing = dislike_timestamps.get(item_id)
            if existing is None or ts > existing:
                dislike_timestamps[item_id] = ts

        recent_dislike_ids: set[UUID] = set()
        decayed_dislike_penalties: dict[UUID, float] = {}

        for item_id, ts in dislike_timestamps.items():
            if ts >= hard_cutoff:
                recent_dislike_ids.add(item_id)
            else:
                days_since = (datetime.utcnow() - ts).days
                penalty = 0.6 * (0.5 ** (days_since / DISLIKE_DECAY_HALF_LIFE_DAYS))
                decayed_dislike_penalties[item_id] = penalty

        cutoff_order = datetime.utcnow() - timedelta(days=ORDER_PENALTY_DAYS)
        recent_orders = {
            str(order.item_id): order
            for order in order_history
            if order.ordered_at >= cutoff_order
        }

        excluded_names: list[str] = []
        surviving: list[MenuItem] = []

        for item in items:
            if item.id in recent_dislike_ids:
                excluded_names.append(item.name)
                continue
            surviving.append(item)

        scored: list[tuple[MenuItem, float]] = []
        for item in surviving:
            history = interaction_history.get(item.id)
            novelty = self.interaction_history_service.calculate_novelty_bonus(history)

            old_dislike_penalty = decayed_dislike_penalties.get(item.id, 0.0)

            order_penalty = 0.0
            item_id_str = str(item.id)
            if item_id_str in recent_orders:
                order = recent_orders[item_id_str]
                days_ago = (datetime.utcnow() - order.ordered_at).days
                order_penalty = 0.3 * (1.0 - days_ago / ORDER_PENALTY_DAYS)
                if order.enjoyed:
                    order_penalty *= 0.5

            combined = novelty - order_penalty - old_dislike_penalty
            scored.append((item, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "Cross-session filters applied",
            extra={
                "initial_count": len(items),
                "hard_excluded_count": len(excluded_names),
                "hard_excluded_names": excluded_names[:10],
                "decayed_dislike_count": len(decayed_dislike_penalties),
                "final_count": len(scored),
                "recently_ordered_count": len(recent_orders),
            },
        )

        return [item for item, _ in scored]

    def _prioritize_fresh_candidates(
        self,
        candidates: list[MenuItem],
        interaction_history: Dict,
        min_candidates: int = 20,
    ) -> list[MenuItem]:
        fresh: list[MenuItem] = []
        seen_not_ordered: list[MenuItem] = []
        ordered: list[MenuItem] = []

        for item in candidates:
            history = interaction_history.get(item.id)
            if not history:
                fresh.append(item)
            elif history.was_ordered:
                ordered.append(item)
            else:
                seen_not_ordered.append(item)

        if len(fresh) >= min_candidates:
            result = fresh
        else:
            result = fresh + seen_not_ordered
            if len(result) < min_candidates:
                result = result + ordered

        logger.info(
            "Fresh-first candidate prioritization",
            extra={
                "total_candidates": len(candidates),
                "fresh_count": len(fresh),
                "seen_not_ordered_count": len(seen_not_ordered),
                "ordered_count": len(ordered),
                "final_count": len(result),
                "fresh_only": len(fresh) >= min_candidates,
            },
        )

        return result

    def _format_item_response(
        self,
        db_session: Session,
        item: MenuItem,
        user: User,
        recommendation_session: RecommendationSession,
        base_scores: Dict[str, float],
        order_history: List[UserOrderHistory],
        user_interaction_history: Dict,
        restaurant_has_breakfast: bool = True
    ) -> Dict[str, Any]:
        score = base_scores.get(str(item.id), 0.5)
        
        confidence, confidence_explanation = self.confidence_service.calculate_recommendation_confidence(
            db_session=db_session,
            user=user,
            item=item,
            recommendation_session=recommendation_session,
            base_score=score
        )
        
        novelty_indicator = self.confidence_service.get_novelty_indicator(
            user_interaction_history.get(item.id)
        )
        
        user_sorted = sorted(user.taste_vector.items(), key=lambda kv: kv[1], reverse=True)
        matched = [k for k, v in user_sorted if item.features.get(k, 0.0) > 0.5][:3]
        
        ranking_factors = {
            "taste_similarity": score,
            "course_match": 1.0 if item.course else 0.5
        }
        
        explanation = self.explanation_enhancement.generate_personalized_explanation(
            item,
            user,
            recommendation_session,
            ranking_factors,
            order_history,
            confidence,
            restaurant_has_breakfast
        )
        
        item_history = user_interaction_history.get(item.id)
        times_seen_before = item_history.times_shown if item_history else 0
        
        return {
            "item_id": str(item.id),
            "name": item.name,
            "description": item.description,
            "course": item.course,
            "price": item.price,
            "image_url": item.provenance.get("image_url") if item.provenance else None,
            "score": round(score, 3),
            "confidence": round(confidence, 2),
            "confidence_explanation": confidence_explanation,
            "novelty_indicator": novelty_indicator,
            "times_seen_before": times_seen_before,
            "explanation": explanation,
            "matched_axes": matched,
            "cuisine": item.cuisine,
            "dietary_tags": item.dietary_tags,
            "features": item.features,
            "safety_confidence": 1.0
        }
    
    def _calculate_ingredient_penalty(self, user: User, item: MenuItem) -> float:
        if not hasattr(user, "ingredient_penalties") or not user.ingredient_penalties:
            return 0.0
        
        if not item.ingredients:
            return 0.0
        
        total_penalty = 0.0
        matching_ingredients = []
        
        # Check item's ingredients against user's learned ingredient penalties
        for ingredient in item.ingredients[:10]:  # Check top 10 ingredients
            ingredient_lower = ingredient.lower().strip()
            if ingredient_lower in user.ingredient_penalties:
                penalty = user.ingredient_penalties[ingredient_lower]
                total_penalty += penalty
                matching_ingredients.append(f"{ingredient_lower}({penalty:.2f})")
        
        if total_penalty > 0:
            logger.debug(
                "Applied ingredient penalty",
                extra={
                    "user_id": str(user.id),
                    "item_id": str(item.id),
                    "item_name": item.name,
                    "total_penalty": round(total_penalty, 3),
                    "matching_ingredients": matching_ingredients
                }
            )
        
        # Scale penalty: each disliked ingredient contributes, but cap at 0.5 total
        return min(0.5, total_penalty * 0.1)

