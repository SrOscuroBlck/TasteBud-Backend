from __future__ import annotations
import logging
import re
from urllib.parse import urljoin

from playwright.async_api import Page, Locator

from .web_scraper import WebScraper, WebScraperError
from .scraping_models import (
    ScrapedMenuItem,
    ScrapedMenuSection,
    ScrapedMenuData,
    ScrapedMenuResult,
    ScrapingProvider,
)

logger = logging.getLogger(__name__)

MENUPP_BASE_URL = "https://menupp.co"

ITEM_BTN_SELECTOR = "a.item-btn"
MENU_LINK_SELECTOR = "a[href*='/venue/'][href*='/menu/']"
VENUE_LINK_SELECTOR = "a[href*='/venue/']"
GROUP_LINK_SELECTOR = "a[href*='/group/']"

PRODUCT_CARD_SELECTOR = ".menu-product-card"
PRODUCT_TITLE_SELECTOR = ".menu-product-title"
CATEGORY_TITLE_SELECTOR = ".menu-category-title"
BADGE_SELECTOR = ".q-badge"
PRODUCT_IMAGE_SELECTOR = ".q-img__container img"
PRICE_BOLD_SELECTOR = "span.text-bold"
PRICE_LABEL_SELECTOR = ".price-label"


class MenuppScraperError(WebScraperError):
    pass


class MenuppScraper:
    def __init__(self, headless: bool = True):
        self._headless = headless

    async def scrape(self, url: str) -> ScrapedMenuResult:
        if not url:
            raise MenuppScraperError("URL is required to scrape a Menupp menu")

        normalized = self._normalize_menupp_url(url)
        entry_type = self._determine_entry_type(normalized)

        async with WebScraper(headless=self._headless) as browser:
            restaurant_name, menu_links = await self._discover_menus(
                browser, normalized, entry_type
            )

            if not menu_links:
                raise MenuppScraperError(f"No menu links discovered from {normalized}")

            menus: list[ScrapedMenuData] = []
            total_items = 0

            for menu_name, menu_url in menu_links:
                menu_data = await self._scrape_single_menu(browser, menu_name, menu_url)
                menus.append(menu_data)
                total_items += sum(len(s.items) for s in menu_data.sections)

            if not restaurant_name:
                restaurant_name = self._extract_restaurant_slug(normalized)

            return ScrapedMenuResult(
                restaurant_name=restaurant_name,
                restaurant_location=None,
                source_url=normalized,
                provider=ScrapingProvider.MENUPP,
                menus=menus,
                total_items=total_items,
            )

    def _normalize_menupp_url(self, url: str) -> str:
        url = url.strip().rstrip("/")

        if not url.startswith("http"):
            url = f"https://{url}"

        if "menupp.co" not in url:
            raise MenuppScraperError(f"URL does not appear to be a Menupp URL: {url}")

        return url

    def _determine_entry_type(self, url: str) -> str:
        if "/venue/" in url and "/menu/" in url:
            return "direct_menu"
        if "/venue/" in url:
            return "venue"
        if "/group/" in url:
            return "group"
        return "home"

    async def _discover_menus(
        self, browser: WebScraper, url: str, entry_type: str
    ) -> tuple[str | None, list[tuple[str, str]]]:
        if entry_type == "direct_menu":
            return None, [("Menu", url)]

        if entry_type == "venue":
            return await self._discover_from_venue_page(browser, url)

        if entry_type == "group":
            return await self._discover_from_group_page(browser, url)

        return await self._discover_from_home_page(browser, url)

    async def _discover_from_group_page(
        self, browser: WebScraper, group_url: str
    ) -> tuple[str | None, list[tuple[str, str]]]:
        page = await browser.open_page(
            group_url,
            wait_selector=MENU_LINK_SELECTOR,
            wait_after_load_ms=5_000,
        )
        try:
            restaurant_name = await self._extract_restaurant_name_from_title(page)
            menu_links = await self._extract_menu_links(page)
            return restaurant_name, menu_links
        finally:
            await page.context.close()

    async def _discover_from_venue_page(
        self, browser: WebScraper, venue_url: str
    ) -> tuple[str | None, list[tuple[str, str]]]:
        page = await browser.open_page(
            venue_url,
            wait_selector=f"{MENU_LINK_SELECTOR}, {PRODUCT_CARD_SELECTOR}, {ITEM_BTN_SELECTOR}",
            wait_after_load_ms=5_000,
        )
        try:
            restaurant_name = await self._extract_restaurant_name_from_title(page)

            menu_links = await self._extract_menu_links(page)
            if menu_links:
                return restaurant_name, menu_links

            if await page.locator(PRODUCT_CARD_SELECTOR).count() > 0:
                logger.info("Venue page has product cards directly, treating as single menu")
                return restaurant_name, [("Menu", venue_url)]

            venue_links = await self._extract_venue_links(page)
            if venue_links:
                logger.info("Venue page has %d sub-venue links, following each", len(venue_links))
                all_menu_links: list[tuple[str, str]] = []
                for venue_name, sub_venue_url in venue_links:
                    _, sub_links = await self._discover_from_venue_page(
                        browser, sub_venue_url
                    )
                    all_menu_links.extend(sub_links)
                if all_menu_links:
                    return restaurant_name, all_menu_links

        finally:
            await page.context.close()

        raise MenuppScraperError(
            f"No menus found on venue page {venue_url}"
        )

    async def _discover_from_home_page(
        self, browser: WebScraper, home_url: str
    ) -> tuple[str | None, list[tuple[str, str]]]:
        page = await browser.open_page(
            home_url,
            wait_selector=ITEM_BTN_SELECTOR,
            wait_after_load_ms=5_000,
        )
        try:
            restaurant_name = await self._extract_restaurant_name_from_title(page)

            direct_links = await self._extract_menu_links(page)
            if direct_links:
                return restaurant_name, direct_links

            group_url = await self._find_group_link(page)
            if group_url:
                await page.context.close()
                _, menu_links = await self._discover_from_group_page(browser, group_url)
                return restaurant_name, menu_links

            venue_links = await self._extract_venue_links(page)
            if venue_links:
                logger.info("Home page has %d venue links, following each", len(venue_links))
                all_menu_links: list[tuple[str, str]] = []
                for _, venue_url in venue_links:
                    try:
                        _, sub_links = await self._discover_from_venue_page(
                            browser, venue_url
                        )
                        all_menu_links.extend(sub_links)
                    except MenuppScraperError:
                        logger.warning("No menus in venue %s, skipping", venue_url)
                if all_menu_links:
                    return restaurant_name, all_menu_links

        finally:
            try:
                await page.context.close()
            except Exception:
                pass

        raise MenuppScraperError(
            f"Could not find menu, group, or venue navigation on {home_url}"
        )

    async def _find_group_link(self, page: Page) -> str | None:
        group_anchors = page.locator(GROUP_LINK_SELECTOR)
        if await group_anchors.count() > 0:
            href = await group_anchors.first.get_attribute("href")
            if href:
                return urljoin(MENUPP_BASE_URL, href)

        item_btns = page.locator(ITEM_BTN_SELECTOR)
        count = await item_btns.count()

        for i in range(count):
            btn = item_btns.nth(i)
            text = await btn.text_content()
            if text and "menú" in text.strip().lower():
                href = await btn.get_attribute("href")
                if href:
                    return urljoin(MENUPP_BASE_URL, href)

        return None

    async def _extract_menu_links(self, page: Page) -> list[tuple[str, str]]:
        links: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        anchors = page.locator(MENU_LINK_SELECTOR)
        count = await anchors.count()

        for i in range(count):
            anchor = anchors.nth(i)
            href = await anchor.get_attribute("href")

            if not href:
                continue

            full_url = urljoin(MENUPP_BASE_URL, href)

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            name_el = anchor.locator(".q-item__section--main")
            name = (
                await name_el.text_content() if await name_el.count() > 0 else None
            )
            if not name:
                name = await anchor.text_content()

            clean_name = name.strip() if name else f"Menu {len(links) + 1}"
            links.append((clean_name, full_url))

        return links

    async def _extract_venue_links(self, page: Page) -> list[tuple[str, str]]:
        links: list[tuple[str, str]] = []
        seen_urls: set[str] = set()

        anchors = page.locator(VENUE_LINK_SELECTOR)
        count = await anchors.count()

        for i in range(count):
            anchor = anchors.nth(i)
            href = await anchor.get_attribute("href")

            if not href:
                continue

            if "/menu/" in href:
                continue

            full_url = urljoin(MENUPP_BASE_URL, href)

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            name_el = anchor.locator(".q-item__section--main")
            name = (
                await name_el.text_content() if await name_el.count() > 0 else None
            )
            if not name:
                name = await anchor.text_content()

            clean_name = name.strip() if name else f"Venue {len(links) + 1}"
            links.append((clean_name, full_url))

        return links

    async def _extract_restaurant_name_from_title(self, page: Page) -> str:
        title = await page.title()
        name = title.replace(" - Menüpp", "").strip()
        return name if name else "Unknown Restaurant"

    def _extract_restaurant_slug(self, url: str) -> str:
        cleaned = url.replace("https://", "").replace("http://", "")
        parts = cleaned.split("/")
        if len(parts) >= 2:
            slug = parts[1]
            return slug.replace("-", " ").title()
        return "Unknown Restaurant"

    async def _scrape_single_menu(
        self, browser: WebScraper, menu_name: str, menu_url: str
    ) -> ScrapedMenuData:
        logger.info("Scraping menu: %s at %s", menu_name, menu_url)

        page = await browser.open_page(
            menu_url,
            wait_selector=PRODUCT_CARD_SELECTOR,
            wait_after_load_ms=3_000,
        )

        try:
            await self._dismiss_dialogs(page)
            await self._scroll_all_content(page, browser)
            await self._trigger_image_loading(page)

            sections = await self._extract_sections(page)

            return ScrapedMenuData(
                menu_name=menu_name,
                menu_url=menu_url,
                sections=sections,
            )
        finally:
            await page.context.close()

    async def _dismiss_dialogs(self, page: Page) -> None:
        dialog_backdrop = page.locator(".q-dialog__backdrop")
        if await dialog_backdrop.count() > 0:
            logger.info("Dismissing dialog overlay")
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1_000)

        dialog_close = page.locator(
            ".q-dialog .q-btn--round, .q-dialog [aria-label='Close']"
        )
        if await dialog_close.count() > 0:
            try:
                await dialog_close.first.click(timeout=3_000)
                await page.wait_for_timeout(500)
            except Exception:
                pass

    async def _scroll_all_content(
        self, page: Page, browser: WebScraper
    ) -> None:
        for pass_num in range(3):
            previous_cards = await page.locator(PRODUCT_CARD_SELECTOR).count()
            await browser.scroll_page_to_bottom(page, step_px=400, delay_ms=600)
            await page.wait_for_timeout(2_000)
            current_cards = await page.locator(PRODUCT_CARD_SELECTOR).count()
            logger.info(
                "Scroll pass %d: %d -> %d product cards",
                pass_num + 1,
                previous_cards,
                current_cards,
            )
            if current_cards == previous_cards and pass_num > 0:
                break

    async def _trigger_image_loading(self, page: Page) -> None:
        await page.evaluate("""
        async () => {
            const step = 400;
            const delay = 150;
            for (let y = document.body.scrollHeight; y >= 0; y -= step) {
                window.scrollTo(0, y);
                await new Promise(r => setTimeout(r, delay));
            }
            window.scrollTo(0, 0);
        }
        """)
        await page.wait_for_timeout(2_000)
        loaded = await page.evaluate("""
        () => document.querySelectorAll('.q-img__container img[src*="cloudfront"]').length
        """)
        logger.info("Images loaded after reverse scroll: %d", loaded)

    async def _extract_sections(self, page: Page) -> list[ScrapedMenuSection]:
        category_containers = page.locator("[id^='category-']")
        container_count = await category_containers.count()
        logger.info("Found %d category containers", container_count)

        if container_count == 0:
            items = await self._extract_all_items_flat(page, "General")
            if items:
                return [ScrapedMenuSection(name="General", items=items)]
            return []

        sections: list[ScrapedMenuSection] = []

        for i in range(container_count):
            container = category_containers.nth(i)
            container_id = await container.get_attribute("id") or f"unknown-{i}"
            cards_in_container = await container.locator(PRODUCT_CARD_SELECTOR).count()
            logger.info(
                "Category container %s: %d product cards",
                container_id,
                cards_in_container,
            )

            section_group = await self._extract_section_group(container)
            extracted_count = sum(len(s.items) for s in section_group)
            logger.info(
                "Extracted %d items from %d sections in %s",
                extracted_count,
                len(section_group),
                container_id,
            )
            sections.extend(section_group)

        return sections

    async def _extract_section_group(
        self, container: Locator
    ) -> list[ScrapedMenuSection]:
        category_titles = container.locator(CATEGORY_TITLE_SELECTOR)
        title_count = await category_titles.count()
        category_name = "Uncategorized"
        if title_count > 0:
            raw = await category_titles.first.text_content()
            category_name = self._clean_category_name(raw or "")

        subcategory_blocks = container.locator(":scope > .q-pb-sm")
        sub_count = await subcategory_blocks.count()

        if sub_count > 0:
            sections: list[ScrapedMenuSection] = []
            for j in range(sub_count):
                sub_block = subcategory_blocks.nth(j)
                section = await self._extract_section_from_block(sub_block)
                if section and section.items:
                    sections.append(section)

            if sections:
                return sections

        items = await self._extract_items_from_container(container, category_name)
        if items:
            return [ScrapedMenuSection(name=category_name, items=items)]

        return []

    async def _extract_section_from_block(
        self, block: Locator
    ) -> ScrapedMenuSection | None:
        title_locator = block.locator(CATEGORY_TITLE_SELECTOR).first

        title_text = ""
        if await title_locator.count() > 0:
            raw_title = await title_locator.text_content()
            title_text = self._clean_category_name(raw_title or "")

        if not title_text:
            title_text = "General"

        items = await self._extract_items_from_container(block, title_text)
        if not items:
            return None

        return ScrapedMenuSection(name=title_text, items=items)

    async def _extract_items_from_container(
        self, container: Locator, category: str
    ) -> list[ScrapedMenuItem]:
        cards = container.locator(PRODUCT_CARD_SELECTOR)
        card_count = await cards.count()
        items: list[ScrapedMenuItem] = []

        for k in range(card_count):
            card = cards.nth(k)
            item = await self._parse_product_card(card, category)
            if item:
                items.append(item)

        return items

    async def _extract_all_items_flat(
        self, page: Page, category: str
    ) -> list[ScrapedMenuItem]:
        cards = page.locator(PRODUCT_CARD_SELECTOR)
        card_count = await cards.count()
        items: list[ScrapedMenuItem] = []

        for k in range(card_count):
            card = cards.nth(k)
            item = await self._parse_product_card(card, category)
            if item:
                items.append(item)

        return items

    async def _parse_product_card(
        self, card: Locator, category: str
    ) -> ScrapedMenuItem | None:
        name = await self._extract_text(card, PRODUCT_TITLE_SELECTOR)
        if not name:
            return None

        description = await self._extract_product_description(card)
        image_url = await self._extract_product_image(card)
        prices = await self._extract_prices(card)
        tags = await self._extract_product_tags(card)
        is_recommended = await self._has_badge(card, "Recomendado")
        is_new = await self._has_badge(card, "Nuevo")

        price = prices[0][1] if prices else None
        price_label = prices[0][0] if prices and prices[0][0] else None

        return ScrapedMenuItem(
            name=self._clean_item_name(name),
            description=description.strip() if description else "",
            price=price,
            price_label=price_label,
            image_url=image_url,
            category=category,
            tags=tags,
            is_recommended=is_recommended,
            is_new=is_new,
        )

    async def _extract_text(self, parent: Locator, selector: str) -> str:
        locator = parent.locator(selector).first
        if await locator.count() == 0:
            return ""
        text = await locator.text_content()
        return text.strip() if text else ""

    async def _extract_product_description(self, card: Locator) -> str:
        desc_locator = card.locator(
            "span.menu-product-description, span.ellipsis-2-lines"
        ).first
        if await desc_locator.count() == 0:
            return ""
        text = await desc_locator.text_content()
        return text.strip() if text else ""

    async def _extract_product_image(self, card: Locator) -> str | None:
        img_elements = card.locator(".q-img img, .q-img__container img, img")
        count = await img_elements.count()

        for i in range(count):
            img = img_elements.nth(i)
            for attr in ("src", "data-src", "srcset"):
                value = await img.get_attribute(attr)
                url = self._extract_valid_image_url(value)
                if url:
                    return url

        containers = card.locator(".q-img, .q-img__container")
        container_count = await containers.count()
        for i in range(container_count):
            style = await containers.nth(i).get_attribute("style")
            if style:
                match = re.search(
                    r'url\(["\']?(https?://[^"\'\)\s]+)["\']?\)', style
                )
                if match:
                    url = self._extract_valid_image_url(match.group(1))
                    if url:
                        return url

        return None

    def _extract_valid_image_url(self, value: str | None) -> str | None:
        if not value:
            return None
        url = value.split(",")[0].split()[0].strip()
        if not url or url.startswith("data:"):
            return None
        if "/others/" in url:
            return None
        if "dvzwo3mu4ucsq.cloudfront.net" in url:
            return url
        return None

    async def _extract_prices(
        self, card: Locator
    ) -> list[tuple[str | None, float | None]]:
        price_rows = card.locator("[data-v-a7aa3a18] .row.full-width.wrap")
        row_count = await price_rows.count()
        prices: list[tuple[str | None, float | None]] = []

        for i in range(row_count):
            row = price_rows.nth(i)

            label_locator = row.locator(PRICE_LABEL_SELECTOR)
            label = None
            if await label_locator.count() > 0:
                label_text = await label_locator.text_content()
                label = label_text.strip() if label_text else None

            bold_locator = row.locator(PRICE_BOLD_SELECTOR)
            if await bold_locator.count() == 0:
                continue

            price_text = await bold_locator.first.text_content()
            price_value = self._parse_price(price_text)

            prices.append((label, price_value))

        return prices

    async def _extract_product_tags(self, card: Locator) -> list[str]:
        tags: list[str] = []

        badge_locator = card.locator(BADGE_SELECTOR)
        badge_count = await badge_locator.count()

        for i in range(badge_count):
            badge = badge_locator.nth(i)
            text = await badge.text_content()
            if text and text.strip():
                tags.append(text.strip())

        return tags

    async def _has_badge(self, card: Locator, badge_text: str) -> bool:
        badges = card.locator(BADGE_SELECTOR)
        count = await badges.count()

        for i in range(count):
            text = await badges.nth(i).text_content()
            if text and badge_text.lower() in text.strip().lower():
                return True

        return False

    def _parse_price(self, price_text: str | None) -> float | None:
        if not price_text:
            return None

        cleaned = price_text.strip()
        cleaned = cleaned.replace("$", "").replace("€", "").strip()
        cleaned = re.sub(r"[^\d.,]", "", cleaned)

        if not cleaned:
            return None

        if "." in cleaned and "," in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif "." in cleaned:
            parts = cleaned.split(".")
            if len(parts) == 2 and len(parts[1]) == 3:
                cleaned = cleaned.replace(".", "")
            elif len(parts) > 2:
                cleaned = cleaned.replace(".", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")

        try:
            return float(cleaned)
        except ValueError:
            logger.warning("Could not parse price: %s", price_text)
            return None

    def _clean_category_name(self, raw: str) -> str:
        cleaned = raw.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.title() if cleaned.isupper() else cleaned

    def _clean_item_name(self, raw: str) -> str:
        cleaned = raw.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.title() if cleaned.isupper() else cleaned
