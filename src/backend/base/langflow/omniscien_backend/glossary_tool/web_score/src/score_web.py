import os
import socket
import ssl
import urllib.error
from datetime import datetime, timezone
from urllib.parse import urlparse

import tldextract
import whois
from loguru import logger

from langflow.omniscien_backend.glossary_tool.web_score.src.utility import (
    DuckDuckGoSearch,
    check_redirection,
    search_func_serp,
    wikipedia_tool,
)
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame

TRUSTED_REGISTRARS = ["GoDaddy", "NameCheap", "Google", "Name.com", "Cloudflare"]
TRUSTED_DNS_PROVIDERS = ["cloudflare.com", "google.com", "awsdns", "akamai", "dynect"]
TRUSTED_SSL_ISSUERS = ["DigiCert", "Let's Encrypt", "Google Trust", "Sectigo", "Amazon", "GlobalSign"]


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


def wiki_site_collect(wiki_search_result: list[Data]):
    """Collects data from the Wikipedia site using the Wikipedia tool wrapper.

    Args:
        wiki_search_result (list[Data]): List of Data objects containing search results from Wikipedia.

    Returns:
        list: List of URLs extracted from the search results.
    """
    url_list = [item.data.get("source") for item in wiki_search_result if item.data.get("source")]
    return url_list


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


def duckduck_go_search_func(query: str) -> list[Data]:
    """Perform a DuckDuckGo search and return results as Data objects."""
    try:
        search_dataframe: DataFrame = DuckDuckGoSearch().perform_search(query=query)
        return search_dataframe.to_data_list()
    except Exception:
        return []


def get_whois_data(domain):
    try:
        return whois.whois(domain)
    except Exception as e:
        print(f"[!] WHOIS lookup failed for {domain}: {e}")
        return None


def whois_lookup(whois_data: whois) -> (int, list[str]):
    """Perform a WHOIS lookup on the given URL.

    Args:
        whois_data (str): Whois data as a string or domain name.

    Returns:
        dict: WHOIS information if available, otherwise None.
    """

    def domain_age_in_years(creation_date):
        if not creation_date:
            return 0
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        return (datetime.now() - creation_date).days // 365

    score = 0
    reasons = []
    if not whois_data:
        return 0

    age = domain_age_in_years(whois_data.creation_date)
    if age >= 5:
        score += 3
        reasons.append("Domain age > 5 years")
    elif age >= 2:
        score += 2
        reasons.append("Domain age > 2 years")
    elif age > 0:
        score += 1
        reasons.append("Domain is new but not brand new")

    # 2. Registrar
    registrar = (whois_data.registrar or "").lower()
    if any(trusted.lower() in registrar for trusted in TRUSTED_REGISTRARS):
        score += 1
        reasons.append("Registrar is reputable")

    # 3. Expiration date
    expiration = whois_data.expiration_date
    if isinstance(expiration, list):
        expiration = expiration[0]
    if expiration and (expiration - datetime.now()).days > 365:
        score += 1
        reasons.append("Domain registered for > 1 year ahead")

    # 4. Registrant/Org presence
    registrant = whois_data.get("org") or whois_data.get("registrant_name")
    if registrant and not any(s in str(registrant).lower() for s in ["private", "redacted", "proxy"]):
        score += 2
        reasons.append("Registrant is publicly visible")

    # 5. DNS providers
    dns_list = whois_data.name_servers or []
    dns_string = " ".join(dns_list).lower()
    if any(trusted in dns_string for trusted in TRUSTED_DNS_PROVIDERS):
        score += 1
        reasons.append("DNS is hosted by trusted provider")

    return score, reasons


def search_ranking(event_name: str, reference: str) -> int:
    """Search for the official website of an event and return its ranking.

    Args:
        event_name (str): Event name to search for.
        reference (str): Reference URL to match against the search results.

    Returns:
        int: The ranking of the official website in the search results, or 0 if not found.
    """
    query = f"{event_name} Official Website"
    logger.debug(f"Searching for official website with query: {query}")
    results = duckduck_go_search_func(query)
    if not results:
        logger.debug("DuckDuckGo search returned no results, falling back to SerpAPI.")
        if os.getenv("SEARCH_TOOL") == "serpapi":
            results = search_func_serp(
                query,
                max_results=20,
            )
    ranking = 0
    for index, item in enumerate(results):
        redirect_url = check_redirection(item.data.get("link", ""))
        if redirect_url == reference:
            logger.debug(f"Found official website: {reference}")
            ranking = index + 1
    logger.debug("Ranking result: {}".format(ranking if ranking else "Not found in top results"))
    return ranking


def extract_hostname(url_or_domain: str) -> str:
    """Extract the hostname from a URL or domain string.
    :param url_or_domain: URL or domain string to extract the hostname from.
    :return: str: The extracted hostname.
    """
    parsed = urlparse(url_or_domain)
    return parsed.netloc or parsed.path


def get_root_domain(url_or_domain):
    extracted = tldextract.extract(url_or_domain)
    root_domain = f"{extracted.domain}.{extracted.suffix}"
    return root_domain


def get_ssl_info(domain_or_url: str, port: int = 443) -> dict | None:
    """Retrieve SSL certificate information for a given domain or URL.
    :param domain_or_url: Domain or URL to check SSL certificate.
    :param port: Port to connect to (default is 443 for HTTPS).
    :return: dict: SSL certificate information if available, otherwise None.
    """
    hostname = extract_hostname(domain_or_url)

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                return cert
    except Exception as e:
        print(f"[!] SSL check failed for {hostname}: {e}")
        return None


def score_ssl_authority(cert):
    if not cert:
        return 0, ["No SSL certificate info"]

    score = 0
    reasons = []

    now = datetime.now(timezone.utc)

    # 1. Certificate validity (not expired)
    try:
        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
        not_before = datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z")
        if now < not_after:
            score += 1
            reasons.append("SSL certificate is valid")

            if (not_after - now).days > 180:
                score += 1
                reasons.append("SSL certificate valid for more than 6 months")
        else:
            reasons.append("SSL certificate expired")

    except Exception as e:
        reasons.append(f"Could not parse SSL validity dates: {e}")

    # 2. Trusted issuer
    issuer = str(cert.get("issuer"))
    if any(trusted in issuer for trusted in TRUSTED_SSL_ISSUERS):
        score += 1
        reasons.append("SSL certificate issued by trusted authority")

    # 3. Key size (approximation)
    # Getting public key bit size requires `cryptography` or `pyOpenSSL`, so we’ll skip or stub this here

    return score, reasons


def backlink_check(url_to_check: str) -> (bool, list[str]):
    """Verify if a URL appears in search results that link to Domain.

    Args:
        url_to_check: The URL to check
    Returns:
        A list of URLs found in the search results
    """
    # List to store search result URLs
    root_domain = get_root_domain(url_to_check)

    # Search Google for the URL
    query = f'"intext:{root_domain} -site:{root_domain}"'
    logger.debug(f"Searching for: {query}")
    url_link = []

    try:
        search_result = search_func_serp(query=query)
        url_link = [item.data.get("link") for item in search_result if item.data.get("link")]
        len_search = len(search_result)
        # for result in search_func_serp(query=query):
        if len_search >= 5:  # Threshold can be adjusted
            logger.debug(f"URL appears {len_search} times, which is a good sign!")
            return True, url_link
        logger.debug(f"URL appears {len_search} times, which might not be enough.")
        return False, url_link
    except urllib.error.HTTPError:
        logger.error("HTTPError: Too many requests. try API")
    return False, url_link


def main(query: str = "", list_url: list[str] = None) -> dict:
    """Main function to demonstrate the Wikipedia tool wrapper.

    Args:
        query: The search query.
        list_url: Optional list of URLs to process instead of performing a search.

    Returns:
        dict: A dictionary containing the event name and details of the official websites.
    """
    # results = wrapper_wiki(query)
    # wiki_result_list = results.to_data_list()
    # url_list = [item.data.get("source") for item in wiki_result_list if item.data.get("source")]
    #
    # # make request
    # api_request = APIRequest()
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # tasks = [api_request.make_api_request(method="GET", url=url) for url in url_list]
    # responses = loop.run_until_complete(asyncio.gather(*tasks))
    # loop.close()
    #
    # official_websites = []
    # for url, response in zip(url_list, responses):
    #     if response:
    #         logger.debug(f"Response from {url}:")
    #         linked_candidate = transform_content_get_link(response.get("response", ""))
    #         logger.debug(f"Linked candidate: {linked_candidate}")
    #         official_websites.append(linked_candidate)
    filter_list = []
    details = []
    for url in list_url:
        score = 0
        filter_list.append(url)
        collector = {"URL": url}
        ranking: int = search_ranking(event_name=query, reference=url)
        if ranking and ranking <= 10:  # If the URL appears in the top 10 results
            score += 3  # High rank boosts the score
        collector["score"] = score
        collector["google_search_rank"] = ranking if ranking else "Not found in top 10"

        whois_data = get_whois_data(url)
        try:
            whois_score, reasons = whois_lookup(whois_data)
            collector["score"] += whois_score
        except Exception as e:
            logger.debug(f"Error during WHOIS lookup for {url}: {e}")

        # SSL certificate check
        ssl_info = get_ssl_info(url)
        ssl_score, ssl_reasons = score_ssl_authority(ssl_info)
        collector["score"] += ssl_score

        # Backlink check
        is_backlinks_pass, backlinks = backlink_check(url)
        if is_backlinks_pass:
            collector["score"] += 3
            logger.debug(f"Backlinks found for {url}")
        else:
            logger.debug(f"No backlinks found for {url}")
        collector["backlinks"] = backlinks

        details.append(collector)

    details.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {
        "event_name": query,
        "details": details,
    }
