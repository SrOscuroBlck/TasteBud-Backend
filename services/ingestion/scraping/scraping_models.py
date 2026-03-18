from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class ScrapingProvider(str, Enum):
    MENUPP = "menupp"
    GENERIC = "generic"


class ScrapedMenuItem(BaseModel):
    name: str
    description: str = ""
    price: Optional[float] = None
    price_label: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    is_recommended: bool = False
    is_new: bool = False


class ScrapedMenuSection(BaseModel):
    name: str
    items: list[ScrapedMenuItem] = Field(default_factory=list)


class ScrapedMenuData(BaseModel):
    menu_name: str
    menu_url: str
    sections: list[ScrapedMenuSection] = Field(default_factory=list)


class ScrapedMenuResult(BaseModel):
    restaurant_name: str
    restaurant_location: Optional[str] = None
    source_url: str
    provider: ScrapingProvider
    menus: list[ScrapedMenuData] = Field(default_factory=list)
    total_items: int = 0
