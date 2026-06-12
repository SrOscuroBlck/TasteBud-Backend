#!/usr/bin/env python
"""
Backfill content_translations for all MenuItems missing them.

Usage (inside the API container):
    PYTHONPATH=/app python scripts/maintenance/translate_menu_descriptions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlmodel import Session, select

import services.core  # noqa: F401 — establishes import order
from config.database import engine
from models.restaurant import MenuItem
from services.features.translation_service import TranslationService


def main():
    with Session(engine) as session:
        items = session.exec(select(MenuItem)).all()
        print(f"[INFO] {len(items)} menu items found")
        updated = TranslationService().translate_items(items, session)
        print(f"[OK] translations written for {updated} items")


if __name__ == "__main__":
    main()
