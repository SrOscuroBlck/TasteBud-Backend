"""Helpers for serving localized dish content.

English originals live in MenuItem.description / MenuItem.ingredients;
translations live in MenuItem.content_translations keyed by language code.
Always fall back to the original when a translation is missing.
"""
from typing import List, Optional, Tuple

SUPPORTED_CONTENT_LANGS = ("es",)


def localized_content(item, lang: Optional[str]) -> Tuple[str, List[str]]:
    """Returns (description, ingredients) in the requested language."""
    description = item.description or ""
    ingredients = item.ingredients or []

    if not lang or lang.startswith("en"):
        return description, ingredients

    translations = (item.content_translations or {}).get(lang.split("-")[0], {})
    return (
        translations.get("description") or description,
        translations.get("ingredients") or ingredients,
    )


def localized_description(item, lang: Optional[str]) -> str:
    return localized_content(item, lang)[0]
