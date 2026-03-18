import asyncio
import logging

from services.ingestion.scraping.web_scraper import WebScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def debug_page() -> None:
    async with WebScraper(headless=True) as browser:
        page = await browser.open_page(
            "https://menupp.co/odiseo",
            wait_after_load_ms=8_000,
        )

        await page.screenshot(path="/app/data/debug_home.png", full_page=True)
        logger.info("Screenshot saved to /app/data/debug_home.png")

        all_links = page.locator("a")
        count = await all_links.count()
        logger.info("Found %d <a> tags", count)

        for i in range(count):
            link = all_links.nth(i)
            href = await link.get_attribute("href")
            text = await link.text_content()
            clean_text = text.strip()[:60] if text else "(empty)"
            logger.info("  [%d] href=%s  text=%s", i, href, clean_text)

        all_buttons = page.locator("button, .q-btn, [role='button']")
        btn_count = await all_buttons.count()
        logger.info("Found %d buttons/q-btn", btn_count)

        for i in range(min(btn_count, 20)):
            btn = all_buttons.nth(i)
            text = await btn.text_content()
            clean = text.strip()[:60] if text else "(empty)"
            logger.info("  btn[%d] text=%s", i, clean)

        router_links = page.locator("[to], .q-item--clickable, .q-btn")
        rl_count = await router_links.count()
        logger.info("Found %d router/clickable elements", rl_count)

        for i in range(min(rl_count, 20)):
            el = router_links.nth(i)
            to = await el.get_attribute("to")
            text = await el.text_content()
            clean = text.strip()[:60] if text else "(empty)"
            logger.info("  clickable[%d] to=%s text=%s", i, to, clean)

        await page.context.close()


asyncio.run(debug_page())
