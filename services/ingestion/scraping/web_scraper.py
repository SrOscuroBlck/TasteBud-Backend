from __future__ import annotations
import logging
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page, BrowserContext


logger = logging.getLogger(__name__)

BROWSER_TIMEOUT_MS = 60_000
DEFAULT_NAVIGATION_TIMEOUT_MS = 60_000
DEFAULT_WAIT_AFTER_LOAD_MS = 3_000


class WebScraperError(Exception):
    pass


class WebScraper:
    def __init__(self, headless: bool = True):
        self._headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self) -> WebScraper:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None

    async def open_page(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        wait_after_load_ms: int = DEFAULT_WAIT_AFTER_LOAD_MS,
    ) -> Page:
        if not self._browser:
            raise WebScraperError("Browser not started. Call start() first.")

        if not url:
            raise WebScraperError("URL is required to open a page")

        context: BrowserContext = await self._browser.new_context(
            viewport={"width": 430, "height": 932},
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Mobile/15E148 Safari/604.1"
            ),
        )
        page = await context.new_page()
        page.set_default_timeout(BROWSER_TIMEOUT_MS)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_NAVIGATION_TIMEOUT_MS)
        except Exception as e:
            await context.close()
            raise WebScraperError(f"Failed to navigate to {url}: {e}")

        if wait_selector:
            try:
                await page.wait_for_selector(wait_selector, timeout=BROWSER_TIMEOUT_MS)
            except Exception:
                logger.warning("Selector %s not found on %s, proceeding anyway", wait_selector, url)

        if wait_after_load_ms > 0:
            await page.wait_for_timeout(wait_after_load_ms)

        return page

    async def scroll_page_to_bottom(self, page: Page, step_px: int = 800, delay_ms: int = 500) -> None:
        previous_height = 0
        while True:
            current_height = await page.evaluate("document.body.scrollHeight")
            if current_height == previous_height:
                break
            previous_height = current_height
            await page.evaluate(f"window.scrollBy(0, {step_px})")
            await page.wait_for_timeout(delay_ms)

        final_height = await page.evaluate("document.body.scrollHeight")
        if final_height > previous_height:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(delay_ms)
