from __future__ import annotations
from datetime import datetime, timedelta

from models import MenuItem, Restaurant, UserOrderHistory
from models.session import MealIntent
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ContextEnhancementService:
    def apply_hard_time_filters(
        self,
        items: list[MenuItem],
        hour: int,
        strict: bool = True
    ) -> list[MenuItem]:
        if not strict:
            return items

        breakfast_courses = ["breakfast", "brunch"]
        is_breakfast_window = 5 <= hour < 10

        if is_breakfast_window:
            has_breakfast = any(
                (item.course or "").lower() in breakfast_courses
                for item in items
            )
            if has_breakfast:
                filtered = [
                    item for item in items
                    if (item.course or "").lower() in breakfast_courses
                ]
                logger.info(
                    "Breakfast time filter applied",
                    extra={"hour": hour, "before": len(items), "after": len(filtered)}
                )
                return filtered

            logger.info(
                "Breakfast window but restaurant has no breakfast items, falling through to main courses",
                extra={"hour": hour, "item_count": len(items)}
            )

        return items
    
    def apply_meal_intent_filters(
        self,
        items: list[MenuItem],
        meal_intent: str,
        hunger_level: str
    ) -> list[MenuItem]:
        course_mapping = {
            "full_meal": ["appetizer", "starter", "salad", "soup", "main", "entree", "dinner", "dessert", "sweet"],
            "appetizer_only": ["appetizer", "starter", "salad", "soup"],
            "main_only": ["main", "entree", "dinner"],
            "dessert_only": ["dessert", "sweet"],
            "beverage_only": ["beverage", "drink"],
            "light_snack": ["appetizer", "snack", "side", "small plate"]
        }

        allowed_courses = course_mapping.get(meal_intent, [])

        if not allowed_courses:
            logger.warning(
                "Unknown meal_intent, returning all items",
                extra={"meal_intent": meal_intent}
            )
            return items

        filtered = []
        for item in items:
            course = (item.course or "").lower()

            if not course:
                if meal_intent == "full_meal":
                    filtered.append(item)
                continue

            for allowed in allowed_courses:
                if allowed in course:
                    filtered.append(item)
                    break

        if hunger_level == "light":
            filtered = [
                item for item in filtered
                if item.course and "main" not in item.course.lower()
            ]

        if not filtered and items:
            logger.warning(
                "Meal intent filter excluded all items",
                extra={
                    "meal_intent": meal_intent,
                    "original_count": len(items),
                    "available_courses": list(set(
                        (item.course or "unknown").lower() for item in items
                    )),
                }
            )

        return filtered
    
    def apply_repeat_penalty(
        self,
        items: list[MenuItem],
        order_history: list[UserOrderHistory],
        days_threshold: int = 30
    ) -> list[tuple[MenuItem, float]]:
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        recent_orders = {
            str(order.item_id): order
            for order in order_history
            if order.ordered_at >= cutoff_date
        }
        
        scored_items = []
        
        for item in items:
            item_id = str(item.id)
            
            if item_id in recent_orders:
                order = recent_orders[item_id]
                days_ago = (datetime.utcnow() - order.ordered_at).days
                
                penalty = 0.3 * (1.0 - days_ago / days_threshold)
                penalty *= 0.5 if order.enjoyed else 1.0
                
                scored_items.append((item, -penalty))
            else:
                scored_items.append((item, 0.0))
        
        return scored_items
    
    def detect_restaurant_type(
        self,
        restaurant: Restaurant,
        menu_items: list[MenuItem]
    ) -> str:
        if not menu_items:
            return "casual"
        
        avg_price = sum(item.price for item in menu_items if item.price) / len(menu_items)
        
        tags = [tag.lower() for tag in restaurant.tags]
        
        if avg_price > 40 or "fine dining" in tags or "michelin" in tags:
            return "fine_dining"
        
        if any(tag in tags for tag in ["chain", "franchise", "fast"]):
            return "chain"
        
        ethnic_markers = ["japanese", "italian", "mexican", "thai", "indian", "chinese", "korean", "french"]
        if any(marker in tags for marker in ethnic_markers):
            return "ethnic"
        
        if avg_price < 15:
            return "fast_casual"
        
        return "casual"
    
    def get_recommendation_strategy(
        self,
        restaurant_type: str,
        user_experience_level: str
    ) -> str:
        if user_experience_level == "new":
            return "safe_popular"
        
        if restaurant_type == "fine_dining":
            if user_experience_level == "established":
                return "adventurous"
            return "balanced"
        
        if restaurant_type == "fast_casual" or restaurant_type == "chain":
            return "safe_popular"
        
        if restaurant_type == "ethnic" and user_experience_level == "established":
            return "adventurous"
        
        return "balanced"
    
    def separate_by_course(
        self,
        items: list[MenuItem]
    ) -> dict[str, list[MenuItem]]:
        by_course = {
            "appetizer": [],
            "main": [],
            "dessert": [],
            "beverage": [],
            "other": []
        }
        
        for item in items:
            course = (item.course or "").lower()
            
            if "appetizer" in course or "starter" in course:
                by_course["appetizer"].append(item)
            elif "main" in course or "entree" in course or "dinner" in course:
                by_course["main"].append(item)
            elif "dessert" in course:
                by_course["dessert"].append(item)
            elif "beverage" in course or "drink" in course:
                by_course["beverage"].append(item)
            else:
                by_course["other"].append(item)
        
        return by_course
    
    def get_available_intents(
        self,
        menu_items: list[MenuItem]
    ) -> list[MealIntent]:
        by_course = self.separate_by_course(menu_items)
        
        has_appetizer = len(by_course["appetizer"]) > 0
        has_main = len(by_course["main"]) > 0
        has_dessert = len(by_course["dessert"]) > 0
        has_beverage = len(by_course["beverage"]) > 0
        has_snack = has_appetizer or len(by_course["other"]) > 0
        
        available: list[MealIntent] = []
        
        if has_main:
            available.append(MealIntent.MAIN_ONLY)
        
        if has_appetizer and has_main and has_dessert:
            available.append(MealIntent.FULL_MEAL)
        
        if has_appetizer:
            available.append(MealIntent.APPETIZER_ONLY)
        
        if has_dessert:
            available.append(MealIntent.DESSERT_ONLY)
        
        if has_beverage:
            available.append(MealIntent.BEVERAGE_ONLY)
        
        if has_snack:
            available.append(MealIntent.LIGHT_SNACK)
        
        if not available and menu_items:
            logger.warning(
                "No intents matched course data, defaulting to main_only",
                extra={"menu_count": len(menu_items)}
            )
            available.append(MealIntent.MAIN_ONLY)
        
        return available
