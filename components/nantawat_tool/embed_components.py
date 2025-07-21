"""Embed Jina Tool Component."""

from typing import Any

import requests
from langchain_community.embeddings import JinaEmbeddings
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import Embeddings
from langflow.io import MessageTextInput, Output
from loguru import logger
from pydantic import ConfigDict, model_validator


class EmbedModelJ(JinaEmbeddings):
    """Embed model using Jina AI Embedding API."""

    session: Any  #: :meta private:
    url_embedding: str = "http://172.17.101.36:8086/embed"
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate that auth token exists in environment."""
        session = requests.Session()
        session.headers.update({"Content-type": "application/json"})
        values["session"] = session
        return values

    def _embed(self, text: Any) -> list[float]:
        # Call Jina AI Embedding API
        url_jina = "http://172.17.101.36:8086/embed"
        payload = {"inputs": text}
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url_jina, json=payload, headers=headers)
        embeddings = resp.json()
        if embeddings is None or not isinstance(embeddings, list):
            logger.error("Received invalid response from Jina API.")
            msg = "Invalid response from Jina API. Expected a list of embeddings."
            raise ValueError(msg)

        logger.debug("That's it, beautiful and simple logging!")
        logger.debug(f"Received {len(embeddings)} embeddings from Jina API.")
        logger.debug(f"Embeddings: {embeddings}")
        return embeddings[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Call out to Jina's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        for text in texts:
            if not isinstance(text, str):
                msg = f"Expected a string, got {type(text)}"
                raise TypeError(msg)
            response = self._embed(text)
            results.append(response)
        return results

    def embed_query(self, text: str) -> list[float]:
        """Call out to Jina's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed(text)


class EmbedJina(LCModelComponent):
    """Embed Jina Tool Component."""

    display_name = "Embed Jina Tool"
    description = "Make HTTP requests using URL or cURL commands Suck."
    icon = "Globe"
    name = "Embed Jina Tool"

    inputs = [MessageTextInput(name="url_embed", display_name="URL", advanced=False, info="Embedding Url")]
    outputs = [
        Output(display_name="Embeddings", name="embeddings", method="build_embeddings"),
    ]

    def build_embeddings(self) -> Embeddings:
        """Build the embeddings model."""
        if not self.url_embed:
            msg = "Embedding URL is required."
            raise ValueError(msg)
        return EmbedModelJ(session=requests.Session(), url_embedding=self.url_embed)
