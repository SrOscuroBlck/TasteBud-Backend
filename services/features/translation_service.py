"""Translates user-facing dish content (descriptions, ingredients) so the API
can serve each user's language directly. English originals are never modified —
they feed safety filtering and feature extraction.

Called best-effort at ingestion time and by
scripts/maintenance/translate_menu_descriptions.py for backfills.
"""
from __future__ import annotations
import json
from typing import List

from sqlmodel import Session

from models import MenuItem
from services.features.gpt_helper import _client
from utils.localization import SUPPORTED_CONTENT_LANGS
from utils.logger import setup_logger

logger = setup_logger(__name__)

BATCH_SIZE = 20

LANG_NAMES = {"es": "Spanish"}


class TranslationService:
    def translate_items(
        self,
        items: List[MenuItem],
        session: Session,
        target_langs: tuple = SUPPORTED_CONTENT_LANGS,
    ) -> int:
        """Fills content_translations for items missing them. Returns the
        number of items updated. Never raises — translation must not break
        ingestion."""
        updated = 0
        for lang in target_langs:
            pending = [
                item for item in items
                if (item.description or item.ingredients)
                and not (item.content_translations or {}).get(lang)
            ]
            for start in range(0, len(pending), BATCH_SIZE):
                batch = pending[start:start + BATCH_SIZE]
                try:
                    translations = self._translate_batch(batch, lang)
                except Exception as e:
                    logger.error(
                        "Menu content translation failed; items stay in English",
                        extra={"error": str(e), "lang": lang, "batch_size": len(batch)},
                    )
                    continue
                for item, translation in zip(batch, translations):
                    if not translation:
                        continue
                    merged = dict(item.content_translations or {})
                    merged[lang] = translation
                    item.content_translations = merged
                    session.add(item)
                    updated += 1
            session.commit()
        return updated

    def _translate_batch(self, items: List[MenuItem], lang: str) -> List[dict]:
        lang_name = LANG_NAMES.get(lang, lang)
        payload = [
            {"description": item.description or "", "ingredients": item.ingredients or []}
            for item in items
        ]
        prompt = f"""Translate the following restaurant dish content from English to {lang_name}.
Keep dish-specific proper nouns as-is. Use natural, appetizing menu language.
Ingredients must stay short (1-3 words each), lowercase.

Input (JSON array, one object per dish):
{json.dumps(payload, ensure_ascii=False)}

Respond with ONLY a JSON array of the same length and order, where each element is:
{{"description": "<translated description>", "ingredients": ["<translated ingredient>", ...]}}"""

        response = _client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        if not isinstance(result, list) or len(result) != len(items):
            raise ValueError(f"translation batch shape mismatch: got {len(result) if isinstance(result, list) else type(result)}, expected {len(items)}")
        return result
