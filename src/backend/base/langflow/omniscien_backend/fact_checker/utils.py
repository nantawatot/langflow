import asyncio
from urllib.parse import quote

import requests
import trafilatura
from playwright.async_api import async_playwright


async def retrieve_url_content(url: str) -> str:
    """Retrieve content from a URL using multiple strategies.
    Returns the extracted content as a string.
    """
    # Strategy 1: Direct fetch with Trafilatura
    try:
        print(f"Attempting direct fetch for: {url}")
        loop = asyncio.get_event_loop()
        raw_html = await loop.run_in_executor(None, trafilatura.fetch_url, url)
        if raw_html:
            content = await extract_with_trafilatura(raw_html, url)
            if content:
                print(f"Success (Direct Fetch): {url}")
                return content
    except Exception as e:
        print(f"Direct fetch failed for {url}: {e}")

    # Strategy 2: Wayback Machine (Internet Archive)
    print(f"Trying Wayback Machine for: {url}")
    archived_url = await try_wayback_machine(url)
    if archived_url:
        try:
            loop = asyncio.get_event_loop()
            raw_html = await loop.run_in_executor(None, trafilatura.fetch_url, archived_url)
            if raw_html:
                content = await extract_with_trafilatura(raw_html, archived_url)
                if content:
                    print(f"Success (Wayback Machine): {url}")
                    return f"[Internet Archive] {content}"
        except Exception as e:
            print(f"Wayback Machine fetch failed for {archived_url}: {e}")

    # Strategy 3: Browser Automation with Playwright
    print(f"Trying Browser Automation for: {url}")
    try:
        content, success = await try_browser_automation(url)
        if success:
            print(f"Success (Browser Automation): {url}")
            return content
    except Exception as e:
        print(f"Browser automation failed for {url}: {e}")

    # If all strategies fail
    raise Exception(f"Failed to retrieve content after all strategies for: {url}")


async def extract_with_trafilatura(raw_html: str, url: str) -> str | None:
    """Extract content using Trafilatura with multiple strategies."""
    loop = asyncio.get_event_loop()
    # First attempt with precision-focused settings
    extracted_content = await loop.run_in_executor(
        None,
        lambda: trafilatura.extract(raw_html, url=url, include_tables=True, favor_precision=True, output_format="txt"),
    )
    if extracted_content and len(extracted_content.strip()) > 50:
        return extracted_content.strip()
    return None


async def try_wayback_machine(original_url: str) -> str | None:
    """Try to find archived version of the URL in Internet Archive's Wayback Machine."""
    try:
        wayback_api_url = f"https://archive.org/wayback/available?url={quote(original_url)}"
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(wayback_api_url, timeout=10))

        if response.status_code == 200:
            data = response.json()
            archived_snapshots = data.get("archived_snapshots", {})
            closest = archived_snapshots.get("closest", {})

            if closest.get("available") and closest.get("url"):
                archived_url = closest["url"]
                print(f"Found archived version of {original_url}: {archived_url}")
                return archived_url
    except Exception as e:
        print(f"Error querying Wayback Machine for {original_url}: {e}")
    return None


async def try_browser_automation(url: str) -> tuple[str | None, bool]:
    """Use browser automation for JavaScript-heavy sites."""
    try:
        print(f"Browser: Trying Stealth Headless for: {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            html_content = await page.content()
            await browser.close()

            if html_content and len(html_content.strip()) > 500:
                print("Browser: Extracting content with trafilatura...")
                loop = asyncio.get_event_loop()
                extracted_content = await loop.run_in_executor(
                    None,
                    lambda: trafilatura.extract(
                        html_content, url=url, include_tables=True, favor_recall=True, output_format="txt"
                    ),
                )
                if extracted_content and len(extracted_content.strip()) > 50:
                    return f"[Browser Automation] {extracted_content.strip()}", True
    except Exception as e:
        print(f"Browser: Stealth Headless failed with error: {e}")
    return None, False
