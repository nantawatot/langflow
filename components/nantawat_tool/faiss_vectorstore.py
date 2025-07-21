"""FAISS Vector Store Component for Langflow."""

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langflow.custom import Component
from langflow.helpers.data import docs_to_data
from langflow.inputs import FloatInput, QueryInput, StrInput, TabInput
from langflow.io import HandleInput, IntInput
from langflow.schema import DataFrame, dotdict
from langflow.schema.data import Data
from langflow.template import Output
from langflow.utils.component_utils import set_field_display


class FaissVectorStoreComponent(Component):
    """FAISS Vector Store Component for indexing and searching documents."""

    display_name: str = "FAISS_Separate"
    description: str = "FAISS Vector Store with index and search capabilities."
    name = "FAISS_Separate"
    icon = "FAISS"

    inputs = [
        TabInput(
            name="mode",
            display_name="Mode",
            options=["Index", "Search"],
            value="Index",
            real_time_refresh=True,
        ),
        StrInput(
            name="index_name",
            display_name="Index Name",
            value="langflow_index",
            required=True,
        ),
        StrInput(
            name="persist_directory",
            display_name="Persist Directory",
            info="Path to save the FAISS index. Relative to where Langflow is running.",
            required=True,
        ),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        HandleInput(
            name="ingest_data",
            display_name="Ingest Data",
            input_types=["Data", "DataFrame"],
        ),
        QueryInput(
            name="search_query",
            display_name="Search Query",
            info="Enter a query to run a similarity search.",
            placeholder="Enter a query...",
            show=False,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
            show=False,
        ),
        FloatInput(
            name="similarity_threshold",
            display_name="Similarity Threshold",
            info="Minimum similarity score required to return a result.",
            value=0.5,
            show=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Results",
            name="results",
            method="get_results",
        ),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Update the build config based on the selected mode."""
        if field_name == "mode":
            if field_value == "Index":
                set_field_display(build_config=build_config, field="search_query", value=False)
                set_field_display(build_config=build_config, field="number_of_results", value=False)
                set_field_display(build_config=build_config, field="similarity_threshold", value=False)
                set_field_display(build_config=build_config, field="ingest_data", value=True)
            elif field_value == "Search":
                set_field_display(build_config=build_config, field="search_query", value=True)
                set_field_display(build_config=build_config, field="number_of_results", value=True)
                set_field_display(build_config=build_config, field="similarity_threshold", value=True)
                set_field_display(build_config=build_config, field="ingest_data", value=False)

    def get_persist_directory(self) -> Path:
        """Returns the resolved the persist directory path."""
        if not self.persist_directory:
            msg = "Persist Directory must be set."
            raise ValueError(msg)
        # Create a Path object and resolve it to an absolute path
        return Path(self.persist_directory).resolve()

    def _prepare_ingest_data(self) -> list[Any]:
        """Prepares ingest_data by converting DataFrame to Data and handling lists."""
        ingest_data = self.ingest_data
        if not ingest_data:
            return []

        # Ensure ingest_data is a list
        if not isinstance(ingest_data, list):
            ingest_data = [ingest_data]

        prepared_list = []
        for item in ingest_data:
            if isinstance(item, DataFrame):
                prepared_list.extend(item.to_data_list())
            else:
                prepared_list.append(item)
        return prepared_list

    def _get_faiss_index_path(self) -> Path:
        """Returns the path to the FAISS index file."""
        return self.get_persist_directory() / f"{self.index_name}.faiss"

    def _combine_index_with_ingested_data(self, documents: list[Document], faiss_ids: list[int]) -> list[Document]:
        """Combines the FAISS index IDs with the ingested documents.

        by adding the faiss_id to each Document's metadata.
        """
        combined_documents = []

        for doc, faiss_id in zip(documents, faiss_ids, strict=False):
            # Copy the existing metadata and add the FAISS ID
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["faiss_id"] = faiss_id

            # Create a new Document with the same content but updated metadata
            combined_doc = Document(page_content=doc.page_content, metadata=metadata)
            combined_documents.append(combined_doc)

        return combined_documents

    def build_vector_store(self) -> list[Document]:
        """Builds the FAISS index and returns the processed data as a list of Documents."""
        persist_dir = self.get_persist_directory()
        persist_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._get_faiss_index_path()

        if index_path.exists():
            self.log(f"Index already exists at {index_path}. Deleting and rebuilding index...")
            index_path.unlink()

        # Prepare the documents
        prepared_data = self._prepare_ingest_data()
        documents = []
        for _input in prepared_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            elif isinstance(_input, Document):
                documents.append(_input)
            else:
                documents.append(Document(page_content=str(_input), metadata={}))

        if not documents:
            msg = "No documents to index."
            raise ValueError(msg)

        # Embed the documents and build the FAISS index
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding.embed_documents(texts)
        if not embeddings:
            msg = "Embedding function failed to return embeddings."
            raise ValueError(msg)

        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Prepare FAISS ID mapping
        faiss_ids = list(range(len(documents)))
        ids_np = np.array(faiss_ids, dtype=np.int64)

        index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
        index.add_with_ids(embeddings_np, ids_np)

        # Persist the index
        faiss.write_index(index, str(index_path))

        # Bind FAISS IDs to the documents' metadata
        processed_documents = self._combine_index_with_ingested_data(documents, faiss_ids)

        self.log(f"Successfully built and persisted FAISS index for {len(processed_documents)} documents.")
        return processed_documents

    def get_results(self) -> list[Data]:
        """Builds the FAISS index or searches it based on the mode."""
        index_path = self._get_faiss_index_path()

        data: list[Data] = []

        if self.mode == "Index":
            self.log("Building index...")

            documents = self.build_vector_store()

            # Convert the documents to a list of Data objects.
            data = docs_to_data(documents)

        if self.mode == "Search":
            self.log("Searching index...")

            if not index_path.exists():
                msg = "Index not found. Please run the Index mode first."
                raise ValueError(msg)

            if not self.search_query or not isinstance(self.search_query, str) or not self.search_query.strip():
                return []

            index = faiss.read_index(str(index_path))
            query_embedding = self.embedding.embed_query(self.search_query)
            query_vector = np.array([query_embedding], dtype=np.float32)

            k = min(self.number_of_results, index.ntotal)
            if k == 0:
                return []

            distances, indices = index.search(query_vector, k)
            faiss_ids = indices[0].tolist()
            distances = distances[0].tolist()

            # Remove invalid FAISS IDs (optional: sometimes -1 indicates no result) and filter by similarity threshold
            data = [
                Data(data={"faiss_id": faiss_id})
                for faiss_id, distance in zip(faiss_ids, distances, strict=False)
                if faiss_id != -1 and distance > self.similarity_threshold
            ]
        return data
