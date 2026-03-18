import asyncio
import json
import sys
import logging

from services.ingestion.scraping import ScrapingOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://menupp.co/odiseo"


async def run_scraper_test(url: str) -> None:
    orchestrator = ScrapingOrchestrator()

    provider = orchestrator.detect_provider(url)
    logger.info("Detected provider: %s", provider.value)

    result = await orchestrator.scrape_menu(url, headless=True)

    logger.info("Restaurant: %s", result.restaurant_name)
    logger.info("Location: %s", result.restaurant_location)
    logger.info("Menus found (raw): %d", len(result.menus))
    logger.info("Total items (raw): %d", result.total_items)

    result = orchestrator.filter_alcoholic_content(result)

    logger.info("Menus found (filtered): %d", len(result.menus))
    logger.info("Total items (filtered): %d", result.total_items)

    for menu in result.menus:
        logger.info("\n=== %s (%s) ===", menu.menu_name, menu.menu_url)
        for section in menu.sections:
            logger.info("  --- %s (%d items) ---", section.name, len(section.items))
            for item in section.items:
                price_str = f" | ${item.price:,.0f}" if item.price else ""
                image_str = " [IMG]" if item.image_url else ""
                logger.info("    %s%s%s", item.name, price_str, image_str)

    text_output = orchestrator.build_extracted_text(result)
    image_map = orchestrator.build_image_map(result)

    logger.info("\n--- TEXT OUTPUT (first 2000 chars) ---")
    logger.info(text_output[:2000])
    logger.info("\n--- IMAGES FOUND: %d ---", len(image_map))
    for name, url in list(image_map.items())[:10]:
        logger.info("  %s -> %s", name, url[:80])

    output_data = result.dict()
    with open("/app/data/scraper_test_output.json", "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Full output saved to /app/data/scraper_test_output.json")


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    logger.info("Starting scraper test for: %s", url)
    asyncio.run(run_scraper_test(url))
    logger.info("Scraper test completed successfully")


if __name__ == "__main__":
    main()
