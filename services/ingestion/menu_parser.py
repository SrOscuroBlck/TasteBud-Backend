from __future__ import annotations
import json
import logging
from typing import Any, Optional

from config.settings import settings
from models.ingestion import MenuParsingResult, ParsedMenuItem

logger = logging.getLogger(__name__)


class MenuParsingError(Exception):
    pass


class MenuParser:
    def parse_menu_text(
        self,
        extracted_text: str,
        restaurant_name: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> MenuParsingResult:
        if not extracted_text:
            raise ValueError("extracted_text is required to parse menu")

        if not extracted_text.strip():
            raise ValueError("extracted_text cannot be empty")

        if len(extracted_text.strip()) < 50:
            raise ValueError("extracted_text is too short to be a valid menu")

        self.currency = currency
        parsed_data = self._parse_with_llm(extracted_text, restaurant_name, currency)

        return self._validate_and_build_result(parsed_data)

    def _parse_with_llm(
        self,
        menu_text: str,
        restaurant_name: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> dict[str, Any]:
        if not settings.OPENAI_API_KEY:
            raise MenuParsingError("OPENAI_API_KEY is not configured")

        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
        except ImportError:
            raise MenuParsingError("openai package is not installed")

        prompt = self._build_prompt(menu_text, restaurant_name, currency)

        try:
            response = client.responses.create(
                model=settings.OPENAI_MODEL,
                reasoning={"effort": "low"},
                input=[{"role": "user", "content": prompt}],
                text={"format": {"type": "json_object"}},
            )

            content = response.output_text
            if not content:
                raise MenuParsingError("LLM returned empty response")

            parsed = json.loads(content.strip())

            if "menu_items" not in parsed:
                logger.error(
                    "LLM response missing menu_items. Keys: %s, first 500 chars: %s",
                    list(parsed.keys()) if isinstance(parsed, dict) else type(parsed),
                    content[:500],
                )

            return parsed

        except json.JSONDecodeError as e:
            logger.error("JSON parsing failed: %s", e.msg)
            raise MenuParsingError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            if "json" in str(type(e).__name__).lower():
                raise
            raise MenuParsingError(f"LLM parsing failed: {e}")

    def _build_prompt(
        self,
        menu_text: str,
        restaurant_name: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> str:
        parts: list[str] = []

        parts.append(
            "Extract every food item from this restaurant menu into structured JSON."
        )

        if restaurant_name:
            parts.append(f"Restaurant: {restaurant_name}")

        if currency:
            parts.append(f"Prices are in {currency}. Include '{currency}' in the currency field.")

        parts.append("""
Rules:
- Keep dish names in their ORIGINAL LANGUAGE
- If a name is ambiguous without its menu section, prepend context (e.g. "Queso" under "CREPES" → "Crepe de Queso")
- Write descriptions in English
- Infer ingredients, allergens, dietary tags, cuisine, course, cooking method from the dish name and description
- Extract every single dish, skip nothing
- Exclude retail/merchandise items

Return JSON:
{
  "restaurant_name": "string or null",
  "restaurant_location": "string or null",
  "currency": "COP/USD/EUR/etc",
  "menu_items": [
    {
      "name": "dish name in original language",
      "description": "English description",
      "price": 12.99,
      "ingredients": ["ingredient1"],
      "allergens": ["dairy"],
      "dietary_tags": ["vegetarian"],
      "cuisine": ["italian"],
      "spice_level": 0,
      "cooking_method": "grilled",
      "course": "main",
      "inference_confidence": 0.85,
      "raw_text": "original text snippet"
    }
  ],
  "extraction_confidence": 0.90,
  "notes": ""
}""")

        parts.append(f"\nMENU TEXT:\n{menu_text}")

        return "\n\n".join(parts)
    
    def _validate_and_build_result(
        self, parsed_data: dict[str, Any]
    ) -> MenuParsingResult:
        if "menu_items" not in parsed_data:
            raise MenuParsingError("Parsed data missing required field: menu_items")

        raw_items = parsed_data.get("menu_items", [])
        if not isinstance(raw_items, list):
            raise MenuParsingError("menu_items must be a list")

        usd_rate = self._fetch_usd_rate()

        menu_items: list[ParsedMenuItem] = []
        skipped_count = 0

        for item_data in raw_items:
            if not isinstance(item_data, dict):
                continue

            if "name" not in item_data or not item_data["name"]:
                skipped_count += 1
                continue

            try:
                price = self._extract_price(item_data.get("price"), usd_rate)
                if price is None or price <= 0:
                    skipped_count += 1
                    continue

                parsed_item = ParsedMenuItem(
                    name=item_data["name"],
                    description=item_data.get("description", ""),
                    price=price,
                    ingredients=self._normalize_list(item_data.get("ingredients", [])),
                    allergens=self._normalize_list(item_data.get("allergens", [])),
                    dietary_tags=self._normalize_list(item_data.get("dietary_tags", [])),
                    cuisine=self._normalize_list(item_data.get("cuisine", [])),
                    spice_level=self._extract_spice_level(item_data.get("spice_level")),
                    cooking_method=item_data.get("cooking_method"),
                    course=item_data.get("course"),
                    inference_confidence=float(item_data.get("inference_confidence", 0.8)),
                    raw_text=item_data.get("raw_text"),
                )
                menu_items.append(parsed_item)
            except Exception:
                skipped_count += 1
                continue

        if not menu_items:
            raise MenuParsingError("No valid menu items were extracted from the text")

        notes = parsed_data.get("notes", "")
        if skipped_count > 0:
            notes += f" (Skipped {skipped_count} invalid items: no price or zero price)"

        return MenuParsingResult(
            restaurant_name=parsed_data.get("restaurant_name"),
            restaurant_location=parsed_data.get("restaurant_location"),
            menu_items=menu_items,
            extraction_confidence=float(parsed_data.get("extraction_confidence", 0.8)),
            notes=notes,
        )

    def _fetch_usd_rate(self) -> float | None:
        if not hasattr(self, "currency") or not self.currency:
            return None

        if self.currency.upper() == "USD":
            return None

        try:
            import requests

            response = requests.get(
                f"https://api.exchangerate-api.com/v4/latest/{self.currency.upper()}",
                timeout=5,
            )
            response.raise_for_status()
            return response.json()["rates"]["USD"]
        except Exception as e:
            logger.error("Currency conversion API failed: %s", e)
            return None

    def _extract_price(
        self, price_value: Any, usd_rate: float | None
    ) -> float | None:
        if price_value is None:
            return None

        raw_price = 0.0
        if isinstance(price_value, (int, float)):
            raw_price = float(price_value)
        elif isinstance(price_value, str):
            try:
                cleaned = (
                    price_value.replace("$", "")
                    .replace("€", "")
                    .replace("£", "")
                    .replace(",", "")
                    .strip()
                )
                raw_price = float(cleaned)
            except ValueError:
                return None
        else:
            return None

        if usd_rate is not None:
            return round(raw_price * usd_rate, 2)

        return raw_price

    def _extract_spice_level(self, spice_value: Any) -> int | None:
        if spice_value is None:
            return None

        if isinstance(spice_value, int):
            return max(0, min(5, spice_value))

        if isinstance(spice_value, str):
            try:
                level = int(spice_value)
                return max(0, min(5, level))
            except ValueError:
                return None

        return None

    def _normalize_list(self, items: Any) -> list[str]:
        if not isinstance(items, list):
            return []

        return [item.strip().lower() for item in items if isinstance(item, str) and item.strip()]
