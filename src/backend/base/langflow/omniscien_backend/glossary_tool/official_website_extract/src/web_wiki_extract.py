import asyncio

import validators

from langflow.omniscien_backend.glossary_tool.official_website_extract.src.utility import (
    APIRequest,
    check_redirection,
    wikipedia_tool,
)
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


def wrapper_wiki(query: str, top_k: int = 5, doc_content_chars_max: int = 100) -> DataFrame:
    """Wrapper function for the Wikipedia tool.

    Args:
        query (str): The search query for Wikipedia.
        top_k (int): Number of top results to return.
        doc_content_chars_max (int): Maximum characters in document content.

    Returns:
        list: List of documents retrieved from Wikipedia.
    """
    docs = wikipedia_tool(query, top_k=top_k, doc_content_chars_max=doc_content_chars_max)
    data = [Data.from_document(doc) for doc in docs]
    return DataFrame(data)


def transform_content_get_link(web_content: bytes | str) -> str | None:
    """Transform Data object to html.

    Args:
        web_content (bytes | str): The web content to parse.

    Returns:
        Optional[str]: The official website link if found, otherwise None.
    """
    from bs4 import BeautifulSoup

    if isinstance(web_content, bytes):
        web_content = web_content.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(web_content, "html.parser")
    infobox = soup.find("table", class_="infobox")
    if infobox:
        # for b in infobox.find_all("td", class_="infobox-data"):
        for a in infobox.find_all("a", href=True):
            if a["href"].startswith("http") and "official website" in a.get_text().lower():
                return check_redirection(a["href"])
    for a in soup.find_all("a", href=True):
        if "official website" in a.get_text().lower() and a["href"].startswith("http"):
            return check_redirection(a["href"])

    external_links_section = soup.find("span", {"id": "External_links"})
    if external_links_section:
        parent = external_links_section.find_parent("h2")
        if parent:
            for link in parent.find_next("ul").find_all("a", href=True):
                if "official website" in link.get_text().lower() and link["href"].startswith("http"):
                    return check_redirection(link["href"])
    return None


from urllib.parse import urlparse, urlunparse


def normalize_url(url: str, remove_www=True, strip_trailing_slash=True, ignore_query=True, ignore_fragment=True) -> str:
    parsed = urlparse(url)

    # Normalize scheme and netloc (hostname)
    scheme = parsed.scheme.lower()
    netloc = parsed.hostname.lower() if parsed.hostname else ""

    # Remove 'www.' if requested
    if remove_www and netloc.startswith("www."):
        netloc = netloc[4:]

    # Remove port if it's default
    if parsed.port:
        if (scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443):
            pass  # don't add port
        else:
            netloc += f":{parsed.port}"

    # Normalize path
    path = parsed.path
    if strip_trailing_slash and path.endswith("/"):
        path = path.rstrip("/")

    # Optionally ignore query and fragment
    query = "" if ignore_query else parsed.query
    fragment = "" if ignore_fragment else parsed.fragment

    return urlunparse((scheme, netloc, path, "", query, fragment))


def main(query: str = "wikipedia_tool_demo") -> dict:
    """Main function to demonstrate the Wikipedia tool wrapper.

    Args:
        query: The search query.

    Returns:
        dict: A dictionary containing the event name and details of the official websites.
    """
    results = wrapper_wiki(query)
    wiki_result_list = results.to_data_list()
    url_list = [item.data.get("source") for item in wiki_result_list if item.data.get("source")]
    # make request
    api_request = APIRequest()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [api_request.make_api_request(method="GET", url=url) for url in url_list]
    responses = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

    official_websites = []
    for url, response in zip(url_list, responses, strict=False):
        if response:
            # logger.debug(f"Response from {url}:")
            linked_candidate = transform_content_get_link(response.get("response", ""))
            # logger.debug(f"Linked candidate: {linked_candidate}")
            official_websites.append(linked_candidate)

    official_websites = list(
        {normalize_url(url) for url in official_websites if (url is not None) and validators.url(url)}
    )

    return {"url_list_candidate": official_websites}
