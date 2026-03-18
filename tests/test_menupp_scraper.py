import asyncio
import sys
import json

sys.path.insert(0, "/Users/gustavocamargo/Developer/TasteBud/TasteBudBackend")

from services.ingestion.scraping.menupp_scraper import MenuppScraper
from services.ingestion.scraping.scraping_orchestrator import ScrapingOrchestrator


async def test_scrape():
    orchestrator = ScrapingOrchestrator()

    provider = orchestrator.detect_provider("https://menupp.co/odiseo")
    print(f"Detected provider: {provider}")

    scraper = MenuppScraper(headless=True)
    result = await scraper.scrape("https://menupp.co/odiseo")

    print(f"\nRestaurant: {result.restaurant_name}")
    print(f"Location: {result.restaurant_location}")
    print(f"Provider: {result.provider}")
    print(f"Total menus found: {len(result.menus)}")
    print(f"Total items: {result.total_items}")

    for menu in result.menus:
        print(f"\n--- Menu: {menu.menu_name} ({menu.menu_url}) ---")
        print(f"  Sections: {len(menu.sections)}")
        for section in menu.sections:
            print(f"    Section: {section.name} ({len(section.items)} items)")
            for item in section.items[:3]:
                print(f"      - {item.name}: {item.description[:60] if item.description else 'N/A'}... | ${item.price}")
                if item.image_url:
                    print(f"        Image: {item.image_url[:80]}...")
            if len(section.items) > 3:
                print(f"      ... and {len(section.items) - 3} more items")

    text = orchestrator.build_extracted_text(result)
    print(f"\n=== Extracted text length: {len(text)} chars ===")
    print(text[:500])

    images = orchestrator.build_image_map(result)
    print(f"\n=== Image map: {len(images)} images ===")
    for name, url in list(images.items())[:5]:
        print(f"  {name}: {url[:80]}...")


if __name__ == "__main__":
    asyncio.run(test_scrape())
