from typing import Dict, Optional, List, Tuple
import json
from openai import OpenAI
from config.settings import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

TASTE_PROFILE_SYSTEM_PROMPT = """You are a culinary expert analyzing menu items to generate taste profiles.

Generate a JSON response with FOUR components:

1. **taste**: 7 taste dimensions (values 0.0-1.0, only include if ≥ 0.6):
   - sweet: sweetness level
   - sour: sourness/tartness level
   - salty: saltiness level
   - bitter: bitterness level
   - umami: savory/umami level
   - fatty: richness/fat content
   - spicy: heat/spice level (capsaicin, not temperature)

2. **texture**: 3 texture dimensions (values 0.0-1.0, only include if ≥ 0.6):
   - crunchy: crispness/crunchiness
   - creamy: smoothness/creaminess
   - chewy: chewiness/density

3. **richness**: single value 0.0-1.0 for overall heaviness/richness

4. **cuisine_typicality**: for each cuisine listed, rate how representative this dish is (0.0-1.0):
   - 1.0 = quintessential example (e.g., Margherita pizza for Italian)
   - 0.7 = typical representation
   - 0.5 = moderate representation
   - 0.3 = fusion/adapted version
   - 0.0 = misleading categorization

CRITICAL RULES:
- Only include taste/texture axes with strong presence (≥ 0.6)
- Most items have 2-4 dominant taste characteristics
- Separate taste from texture - don't confuse them
- Temperature is NOT a taste (spicy = capsaicin heat, not thermal)
- Return valid JSON only, no explanations

Example for "Chocolate Lava Cake":
{
  "taste": {"sweet": 0.9, "bitter": 0.3, "fatty": 0.8},
  "texture": {"creamy": 0.9},
  "richness": 0.95,
  "cuisine_typicality": {"french": 0.9, "dessert": 1.0}
}

Example for "Thai Green Curry":
{
  "taste": {"spicy": 0.85, "umami": 0.8, "fatty": 0.7, "salty": 0.6},
  "texture": {"creamy": 0.7},
  "richness": 0.75,
  "cuisine_typicality": {"thai": 0.95}
}

Example for "Sushi Burrito":
{
  "taste": {"umami": 0.7, "salty": 0.6},
  "texture": {"chewy": 0.7},
  "richness": 0.5,
  "cuisine_typicality": {"japanese": 0.3, "fusion": 0.9}
}"""


def generate_llm_taste_profile(
    item_name: Optional[str],
    item_description: Optional[str] = None,
    ingredients: Optional[List[str]] = None,
    cuisines: Optional[List[str]] = None
) -> Tuple[Dict[str, float], Dict[str, float], Optional[float], Dict[str, float]]:
    if not item_name:
        return ({}, {}, None, {})
    
    context_parts = [f"Menu Item: {item_name}"]
    
    if item_description and item_description.strip():
        context_parts.append(f"Description: {item_description}")
    
    if ingredients and len(ingredients) > 0:
        ingredients_str = ", ".join(ingredients[:10])
        context_parts.append(f"Ingredients: {ingredients_str}")
    
    if cuisines and len(cuisines) > 0:
        cuisines_str = ", ".join(cuisines)
        context_parts.append(f"Cuisines: {cuisines_str}")
    
    context_text = "\n".join(context_parts)
    context_text += "\n\nReturn valid JSON only."
    
    try:
        response = client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=TASTE_PROFILE_SYSTEM_PROMPT,
            input=context_text,
            reasoning={"effort": "low"},
            text={"format": {"type": "json_object"}},
        )
        
        content = response.output_text.strip()
        
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break
        
        parsed = json.loads(content)
        
        taste = validate_taste_profile(parsed.get("taste", {}))
        texture = validate_texture_profile(parsed.get("texture", {}))
        richness = validate_richness(parsed.get("richness"))
        cuisine_typicality = validate_cuisine_typicality(parsed.get("cuisine_typicality", {}))
        
        return (taste, texture, richness, cuisine_typicality)
        
    except Exception:
        return ({}, {}, None, {})


def validate_taste_profile(taste_data: Dict) -> Dict[str, float]:
    valid_axes = {"sweet", "sour", "salty", "bitter", "umami", "fatty", "spicy"}
    validated = {}
    
    for axis, value in taste_data.items():
        if axis not in valid_axes:
            continue
        
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        
        value = max(0.0, min(1.0, value))
        
        if value >= 0.6 or value <= 0.4:
            validated[axis] = round(value, 3)
    
    return validated


def validate_texture_profile(texture_data: Dict) -> Dict[str, float]:
    valid_axes = {"crunchy", "creamy", "chewy"}
    validated = {}
    
    for axis, value in texture_data.items():
        if axis not in valid_axes:
            continue
        
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        
        value = max(0.0, min(1.0, value))
        
        if value >= 0.6:
            validated[axis] = round(value, 3)
    
    return validated


def validate_richness(richness_value) -> Optional[float]:
    if richness_value is None:
        return None
    
    try:
        value = float(richness_value)
    except (TypeError, ValueError):
        return None
    
    value = max(0.0, min(1.0, value))
    return round(value, 3)


def validate_cuisine_typicality(typicality_data: Dict) -> Dict[str, float]:
    validated = {}
    
    for cuisine, value in typicality_data.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        
        value = max(0.0, min(1.0, value))
        validated[cuisine.lower()] = round(value, 3)
    
    return validated
