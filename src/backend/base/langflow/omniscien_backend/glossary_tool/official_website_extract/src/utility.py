import json
import os
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pandas as pd
import requests
import validators
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_core.documents import Document
from loguru import logger

from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.services.deps import get_settings_service

load_dotenv()


def wikipedia_tool(query: str, language="en", top_k: int = 5, doc_content_chars_max: int = 100) -> list[Document]:
    api_wrapper = WikipediaAPIWrapper(lang=language, top_k_results=top_k, doc_content_chars_max=doc_content_chars_max)
    response = api_wrapper.load(query)
    return response


class DuckDuckGoSearch:
    @staticmethod
    def validate_url(string: str) -> bool:
        url_regex = re.compile(
            r"^(https?:\/\/)?" r"(www\.)?" r"([a-zA-Z0-9.-]+)" r"(\.[a-zA-Z]{2,})?" r"(:\d+)?" r"(\/[^\s]*)?$",
            re.IGNORECASE,
        )
        return bool(url_regex.match(string))

    def ensure_url(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        if not self.validate_url(url):
            msg = f"Invalid URL: {url}"
            raise ValueError(msg)
        return url

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize search query."""
        # Remove potentially dangerous characters
        return re.sub(r'[<>"\']', "", query.strip())

    def perform_search(self, query, timeout: int = 30) -> DataFrame:
        query = self._sanitize_query(query)
        if not query:
            msg = "Empty search query"
            raise ValueError(msg)
        headers = {"User-Agent": get_settings_service().settings.user_agent}
        params = {"q": query, "kl": "us-en"}
        url = "https://html.duckduckgo.com/html/"

        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            return DataFrame(pd.DataFrame([{"title": "Error", "link": "", "snippet": str(e), "content": ""}]))

        if not response.text or "text/html" not in response.headers.get("content-type", "").lower():
            return DataFrame(
                pd.DataFrame([{"title": "Error", "link": "", "snippet": "No results found", "content": ""}])
            )
        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select("div.result"):
            title_tag = result.select_one("a.result__a")
            snippet_tag = result.select_one("a.result__snippet")
            if title_tag:
                raw_link = title_tag.get("href", "")
                parsed = urlparse(raw_link)
                uddg = parse_qs(parsed.query).get("uddg", [""])[0]
                decoded_link = unquote(uddg) if uddg else raw_link

                try:
                    final_url = self.ensure_url(decoded_link)
                    page = requests.get(final_url, headers=headers, timeout=timeout)
                    page.raise_for_status()
                    content = BeautifulSoup(page.text, "lxml").get_text(separator=" ", strip=True)
                except requests.RequestException as e:
                    final_url = decoded_link
                    content = f"(Failed to fetch: {e!s}"

                results.append(
                    {
                        "title": title_tag.get_text(strip=True),
                        "link": final_url,
                        "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                        "content": content,
                    }
                )

        df_results = pd.DataFrame(results)
        return DataFrame(df_results)


class APIRequest:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": get_settings_service().settings.user_agent})

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if the provided URL is well-formed.
        :param url: URL to validate
        :return: True if valid, False otherwise
        """
        if not validators.url(url):
            msg = f"Invalid URL provided: {url}"
            raise ValueError(msg)
        return True

    @staticmethod
    def _parse_json_value(value: Any) -> Any:
        """Parse a value that might be a JSON string."""
        if not isinstance(value, str):
            return value

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        else:
            return parsed

    def _process_dict_body(self, body: dict) -> dict:
        """Process dictionary body by parsing JSON values."""
        return {k: self._parse_json_value(v) for k, v in body.items()}

    def _process_body(self, body: Any) -> dict:
        """Process the body input into a valid dictionary."""
        if body is None:
            return {}
        if isinstance(body, dict):
            return self._process_dict_body(body)
        return {}

    async def make_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict | None = None,
        body: Any = None,
        timeout: int = 5,
        *,
        follow_redirects: bool = True,
    ):
        processed_body = self._process_body(body)
        redirection_history = []

        try:
            request_params = {
                "method": method,
                "url": url,
                "headers": headers,
                "json": processed_body,
                "timeout": timeout,
                "follow_redirects": follow_redirects,
            }
            response = await client.request(**request_params)

            redirection_history = [
                {
                    "url": redirect.headers.get("Location", str(redirect.url)),
                    "status_code": redirect.status_code,
                }
                for redirect in response.history
            ]
            metadata = {
                "source": url,
                "status_code": response.status_code,
                "headers": response.headers,
                "response": response.content,
            }
        except (httpx.HTTPError, httpx.RequestError, httpx.TimeoutException) as exc:
            logger.debug(f"Error making request to {url}")
            logger.error(f"Exception: {exc}")
            return {"error": str(exc), "redirection_history": redirection_history}
        return metadata

    async def make_api_request(
        self,
        method: str,
        url,
        headers: dict | None = None,
        body: Any = None,
        timeout: int = 5,
        follow_redirects=True,
    ):
        """Make an API request to a predefined URL and return the response.
        :return:
        """
        async with httpx.AsyncClient() as client:
            result = await self.make_request(
                client, method, url, headers, body, timeout, follow_redirects=follow_redirects
            )
        return result


def search_func_serp(
    query: str, params: dict[str, Any] | None = None, max_results: int = 5, max_snippet_length: int = 100
) -> list[Data]:
    def _build_wrapper(params: dict[str, Any] | None = None) -> SerpAPIWrapper:
        """Build a SerpAPIWrapper with the provided parameters."""
        params = params or {}
        if params:
            return SerpAPIWrapper(
                serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
                params=params,
            )
        return SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

    wrapper = _build_wrapper()
    try:
        local_wrapper = wrapper
        if params:
            local_wrapper = _build_wrapper(params)

        full_results = local_wrapper.results(query)
        organic_results = full_results.get("organic_results", [])[:max_results]

        limited_results = [
            Data(
                text=result.get("snippet", ""),
                data={
                    "title": result.get("title", "")[:max_snippet_length],
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", "")[:max_snippet_length],
                },
            )
            for result in organic_results
        ]

    except Exception as e:
        error_message = f"Error in SerpAPI search: {e!s}"
        logger.debug(error_message)
        return []

    return limited_results


async def main():
    """Main function to demonstrate the APIRequest class."""
    api_request = APIRequest()
    url = "https://api.github.com/repos/langflow-ai/langflow"
    response = await api_request.make_api_request("GET", url)
    return response


def check_redirection(url):
    """Check if a URL redirects to another URL using a HEAD request.

    Args:
        url (str): The URL to check for redirection.

    Returns:
        str: The final redirected URL after following redirects, or the original URL if no redirection.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except requests.RequestException:
        return url  # In case of error, return the original URL
