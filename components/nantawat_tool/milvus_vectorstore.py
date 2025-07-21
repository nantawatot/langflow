"""Milvus Vector Store Component for Langflow."""

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers.data import docs_to_data
from langflow.io import (
    BoolInput,
    DictInput,
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    Output,
    SecretStrInput,
    StrInput,
    TabInput,
)
from langflow.schema import DataFrame
from langflow.schema.data import Data
from pymilvus import Collection, MilvusClient, db, utility

DIMENSION = 768  # Default dimension for embeddings, can be adjusted based on your use case


class MilvusVectorSTore(LCVectorStoreComponent):
    """Milvus vector store with search capabilities."""

    display_name: str = "MilvusStoreVector"
    description: str = "Milvus vector store with search capabilities"
    name = "MilvusStoreVector"
    icon = "Milvus"

    inputs = [
        TabInput(
            name="mode",
            display_name="Mode",
            options=["Index", "Search"],
            value="Index",
            info="Select the mode for the component. In 'Index' mode, "
            "it will ingest data into the database. "
            "In 'Search' mode, it will run a query against the database.",
            real_time_refresh=True,
        ),
        StrInput(name="db_name", display_name="Database Name", value="langflow_db"),
        StrInput(name="collection_name", display_name="Collection Name", value="langflow"),
        StrInput(name="collection_description", display_name="Collection Description", value=""),
        StrInput(
            name="uri",
            display_name="Connection URI",
            value="http://localhost:19530",
        ),
        SecretStrInput(
            name="password",
            display_name="Token",
            value="",
            info="Ignore this field if no token is required to make connection.",
        ),
        DictInput(name="connection_args", display_name="Other Connection Arguments", advanced=True),
        StrInput(name="primary_field", display_name="Primary Field Name", value="pk"),
        StrInput(name="text_field", display_name="Text Field Name", value="text"),
        StrInput(name="vector_field", display_name="Vector Field Name", value="vector"),
        DropdownInput(
            name="consistency_level",
            display_name="Consistencey Level",
            options=["Bounded", "Session", "Strong", "Eventual"],
            value="Session",
            advanced=True,
        ),
        DictInput(name="index_params", display_name="Index Parameters", advanced=True),
        DictInput(name="search_params", display_name="Search Parameters", advanced=True),
        BoolInput(name="drop_old", display_name="Drop Old Collection", value=False, advanced=True),
        FloatInput(name="timeout", display_name="Timeout", advanced=True),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
            advanced=True,
        ),
    ]
    outputs = [
        Output(
            display_name="Search Results",
            name="search_results",
            method="main",
        )
    ]

    @check_cached_vector_store
    def build_vector_store(self):
        """Builds the Milvus vector store."""
        try:
            from langchain_milvus.vectorstores import Milvus as LangchainMilvus
        except ImportError as e:
            msg = "Could not import Milvus integration package. Please install it with `pip install langchain-milvus`."
            raise ImportError(msg) from e
        self.connection_args.update(uri=self.uri, token=self.password)
        self.initial_client_vector_store()
        return LangchainMilvus(
            embedding_function=self.embedding,
            collection_name=self.collection_name,
            collection_description=self.collection_description,
            connection_args=self.connection_args,
            consistency_level=self.consistency_level,
            index_params=self.index_params,
            search_params=self.search_params,
            drop_old=self.drop_old,
            auto_id=True,
            primary_field=self.primary_field,
            text_field=self.text_field,
            vector_field=self.vector_field,
            timeout=self.timeout,
        )

    def __ingest(self, vector_store):
        """Ingests data into the Milvus vector store."""
        # Convert DataFrame to Data if needed using parent's method
        self.ingest_data = self._prepare_ingest_data()

        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        if documents:
            vector_store.add_documents(documents)

    def query_all(self) -> list[Data]:
        """Run a query against the Milvus vector store."""
        MilvusClient(uri=self.uri, token=self.password)
        collection = Collection(self.collection_name)
        collection.load()
        return collection.query(
            expr="pk > 0",
            output_fields=["pk", "file_path", "text"],
        )

    def search_documents(self) -> list[Data]:
        """Search documents in the Milvus vector store."""
        vector_store = self.build_vector_store()

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            docs = vector_store.similarity_search(
                query=self.search_query,
                k=self.number_of_results,
            )

            data = docs_to_data(docs)
            self.status = data
            return data
        return []

    def main(self) -> list[Data] or DataFrame:
        """Main method to run the Milvus vector store component."""
        vector_store = self.build_vector_store()
        if self.mode == "Index":
            self.__ingest(vector_store)
            res = self.query_all()
            return DataFrame(res)
        return self.search_documents()

    def initial_client_vector_store(self):
        """Initializes the Milvus client and prepares the database and collection."""
        client = MilvusClient(
            uri=self.uri,  # Adjust the URI as needed
            token=self.password,
        )

        existing_databases = db.list_database()
        if self.db_name in existing_databases:
            db.using_database(self.db_name)
        else:
            db.create_database(self.db_name)
            db.using_database(self.db_name)

        collections = utility.list_collections()
        for collection_name in collections:
            collection = Collection(name=collection_name)
            collection.drop()

        client.create_collection(
            collection_name=self.collection_name,
            dimension=DIMENSION,  # The vectors we will use in this demo has 768 dimensions
        )
