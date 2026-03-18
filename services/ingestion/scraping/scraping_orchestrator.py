from __future__ import annotations
import logging
import re
from typing import Optional
from urllib.parse import urlparse

from .scraping_models import (
    ScrapedMenuResult,
    ScrapedMenuData,
    ScrapedMenuSection,
    ScrapedMenuItem,
    ScrapingProvider,
)
from .menupp_scraper import MenuppScraper


logger = logging.getLogger(__name__)

ALCOHOLIC_MENU_PATTERNS = re.compile(
    r"bebidas?\b|c[oó]ctel|cocktail|licor|liquor|liqueur|whisky|whiskey|tequila"
    r"|rones?\b|rums?\b|ginebra|gins?\b|vodka|mezcal|vinos?\b|wines?\b"
    r"|champagne|cervezas?\b|beers?\b|aperitivos?\b|digestivos?\b"
    r"|spirits?\b|bourbon|brandy|cognac|tragos?\b",
    re.IGNORECASE,
)

NON_ALCOHOLIC_PATTERNS = re.compile(
    r"sin alcohol|non.?alcohol|mocktail|virgin|zero|0\.?0|libre de alcohol"
    r"|refrescos|jugos?\b|juices?\b|soft drinks?|gaseosas?|aguas?\b|waters?\b"
    r"|cafés?\b|coffees?\b|tés?\b|teas?\b|limonadas?\b|batidos?\b",
    re.IGNORECASE,
)


class ScrapingOrchestratorError(Exception):
    pass


class ScrapingOrchestrator:
    def detect_provider(self, url: str) -> ScrapingProvider:
        if not url:
            raise ScrapingOrchestratorError("URL is required to detect provider")

        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        if "menupp.co" in hostname:
            return ScrapingProvider.MENUPP

        return ScrapingProvider.GENERIC

    async def scrape_menu(self, url: str, headless: bool = True) -> ScrapedMenuResult:
        if not url:
            raise ScrapingOrchestratorError("URL is required to scrape menu")

        provider = self.detect_provider(url)

        if provider == ScrapingProvider.MENUPP:
            return await self._scrape_menupp(url, headless)

        return await self._scrape_generic(url, headless)

    async def _scrape_menupp(self, url: str, headless: bool) -> ScrapedMenuResult:
        scraper = MenuppScraper(headless=headless)
        return await scraper.scrape(url)

    async def _scrape_generic(self, url: str, headless: bool) -> ScrapedMenuResult:
        raise ScrapingOrchestratorError(
            f"Generic web scraping is not yet implemented. "
            f"Currently supported providers: Menupp (menupp.co). "
            f"URL: {url}"
        )

    def build_extracted_text(self, result: ScrapedMenuResult) -> str:
        lines: list[str] = []
        lines.append(f"Restaurant: {result.restaurant_name}")

        if result.restaurant_location:
            lines.append(f"Location: {result.restaurant_location}")

        lines.append("")

        for menu in result.menus:
            lines.append(f"=== {menu.menu_name} ===")
            lines.append("")

            for section in menu.sections:
                lines.append(f"--- {section.name} ---")

                for item in section.items:
                    item_line = f"  {item.name}"
                    if item.description:
                        item_line += f" - {item.description}"
                    if item.price is not None:
                        item_line += f" | {item.price}"
                    if item.price_label:
                        item_line += f" ({item.price_label})"
                    lines.append(item_line)

                lines.append("")

        return "\n".join(lines)

    def build_image_map(self, result: ScrapedMenuResult) -> dict[str, str]:
        image_map: dict[str, str] = {}

        for menu in result.menus:
            for section in menu.sections:
                for item in section.items:
                    if item.image_url:
                        image_map[item.name] = item.image_url

        return image_map

    def filter_alcoholic_content(
        self, result: ScrapedMenuResult
    ) -> ScrapedMenuResult:
        filtered_menus: list[ScrapedMenuData] = []
        total_items = 0

        for menu in result.menus:
            if self._is_alcoholic_name(menu.menu_name):
                logger.info("Filtering out alcoholic menu: %s", menu.menu_name)
                continue

            filtered_sections: list[ScrapedMenuSection] = []
            for section in menu.sections:
                if self._is_alcoholic_name(section.name):
                    logger.info(
                        "Filtering out alcoholic section: %s", section.name
                    )
                    continue
                filtered_sections.append(section)
                total_items += len(section.items)

            if filtered_sections:
                filtered_menus.append(
                    ScrapedMenuData(
                        menu_name=menu.menu_name,
                        menu_url=menu.menu_url,
                        sections=filtered_sections,
                    )
                )

        return ScrapedMenuResult(
            restaurant_name=result.restaurant_name,
            restaurant_location=result.restaurant_location,
            source_url=result.source_url,
            provider=result.provider,
            menus=filtered_menus,
            total_items=total_items,
        )

    def _is_alcoholic_name(self, name: str) -> bool:
        if NON_ALCOHOLIC_PATTERNS.search(name):
            return False
        return bool(ALCOHOLIC_MENU_PATTERNS.search(name))
