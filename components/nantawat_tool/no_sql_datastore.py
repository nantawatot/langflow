"""NoSQL Datastore Component for LangFlow."""

import tempfile
from typing import Any

import certifi
from langflow.custom.custom_component.component_with_cache import ComponentWithCache
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output, SecretStrInput, StrInput, TabInput
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.services.cache.utils import CacheMiss
from pymongo import MongoClient, UpdateOne
from sqlalchemy.exc import SQLAlchemyError

URI = "mongodb://localhost:27017/"


class MongoDBStore(ComponentWithCache):
    """A sql component."""

    display_name = "NoSQL Database"
    description = "Executes SQL queries on SQLAlchemy-compatible databases."
    icon = "database"
    name = "SQLComponent"
    # metadata = {"keywords": ["sql", "database", "query", "db", "fetch"]}

    inputs = [
        TabInput(
            name="mode",
            display_name="Mode",
            options=["Index", "Search"],
            value="Index",
            info="Select the mode for the component. "
            "In 'Index' mode, it will ingest data into the database. "
            "In 'Search' mode, it will run a query against the database.",
            real_time_refresh=True,
        ),
        MessageTextInput(name="database_url", display_name="Database URL", required=True),
        BoolInput(name="enable_mtls", display_name="Enable mTLS", value=False, advanced=True, required=True),
        SecretStrInput(
            name="mongodb_atlas_client_cert",
            display_name="MongoDB Atlas Combined Client Certificate",
            required=False,
            info="Client Certificate combined with the private key in the following format:\n "
            "-----BEGIN PRIVATE KEY-----\n...\n -----END PRIVATE KEY-----\n-----BEGIN CERTIFICATE-----\n"
            "...\n-----END CERTIFICATE-----\n",
        ),
        StrInput(name="db_name", display_name="Database Name", required=True),
        StrInput(name="collection_name", display_name="Collection Name or Table Name", required=True),
        StrInput(name="column_id", display_name="Column ID", required=True),
        HandleInput(name="query", display_name="SQL Query", input_types=["Data", "DataFrame"]),
        HandleInput(name="ingest_data", display_name="Ingest Data", input_types=["Data", "DataFrame"]),
        IntInput(name="batch_size", display_name="Batch Size", value=1000, advanced=True, required=False),
    ]

    outputs = [
        Output(display_name="Result Table", name="run_sql_query", method="run_sql_query"),
    ]

    def __init__(self, **kwargs) -> None:
        """Initialize the MongoDBStore component."""
        super().__init__(**kwargs)
        self.client: MongoClient | None = None
        self.collection = None

    def maybe_create_db(self):
        """Create the database and collection if they do not exist."""
        if self.database_url != "":
            cached_db = self._shared_component_cache.get(self.database_url)
            if not isinstance(cached_db, CacheMiss):
                self.collection = cached_db[self.db_name][self.collection_name]
                return
            self.log("Connecting to database")
            try:
                self.collection = self.create_client_collection()
            except Exception as e:
                msg = f"An error occurred while connecting to the database: {e}"
                raise ValueError(msg) from e
            self._shared_component_cache.set(self.database_url, self.client)

    def create_client_collection(self):
        """Create a MongoDB client and return the collection."""
        # Create temporary files for the client certificate
        if self.enable_mtls:
            client_cert_path = None
            try:
                client_cert = self.mongodb_atlas_client_cert.replace(" ", "\n")
                client_cert = client_cert.replace("-----BEGIN\nPRIVATE\nKEY-----", "-----BEGIN PRIVATE KEY-----")
                client_cert = client_cert.replace(
                    "-----END\nPRIVATE\nKEY-----\n-----BEGIN\nCERTIFICATE-----",
                    "-----END PRIVATE KEY-----\n-----BEGIN CERTIFICATE-----",
                )
                client_cert = client_cert.replace("-----END\nCERTIFICATE-----", "-----END CERTIFICATE-----")
                with tempfile.NamedTemporaryFile(delete=False) as client_cert_file:
                    client_cert_file.write(client_cert.encode("utf-8"))
                    client_cert_path = client_cert_file.name

            except Exception as e:
                msg = f"Failed to write certificate to temporary file: {e}"
                raise ValueError(msg) from e

        try:
            mongo_client: MongoClient = (
                MongoClient(
                    self.database_url,
                    tls=True,
                    tlsCertificateKeyFile=client_cert_path,
                    tlsCAFile=certifi.where(),
                )
                if self.enable_mtls
                else MongoClient(self.database_url)
            )
            self.client = mongo_client
            collection = mongo_client[self.db_name][self.collection_name]

        except Exception as e:
            msg = f"Failed to connect to MongoDB Atlas: {e}"
            raise ValueError(msg) from e

        return collection

    def __collect_ingest_data(self) -> Data:
        self.log(self.ingest_data)
        return self.ingest_data

    def __delete_db_data(self) -> None:
        """Delete all data in the NoSQL database."""
        self.collection.delete_many({})
        if self.collection.count_documents({}) == 0:
            self.log("All documents deleted successfully.")

    def __ingest(self, *, index: bool = True) -> None:
        """Ingest data into the NoSQL database.

        Data to insert should be provided in the `ingest_data` attribute.
        Format of `ingest_data` can be a list of dictionaries or Data objects.
        [
        {
        text_key: "text",
        data: {
        file_path: "path/to/file",
        faiss_id: "unique_id",
         text: "text content",
         metadata: {"key": "value"}
         },
         default_value: ""
         }
        ].


        """
        self.maybe_create_db()
        self.__delete_db_data()
        if not index:
            self.log("Ingest data is disabled, skipping ingestion.")
            return

        data_ingest = self.__collect_ingest_data()
        batch_size = self.batch_size
        self.log(data_ingest)
        data = [doc.data for doc in data_ingest]
        self.log(data)
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.collection.insert_many(batch)

    def update_with_bulk(self, data: list[dict[str, Any]]) -> None:
        """Update documents in the NoSQL database using bulk operations.

        This method will update documents based on the `faiss_id` field.
        """
        self.maybe_create_db()
        db_store = self.collection

        operations = [UpdateOne({"faiss_id": doc["faiss_id"]}, {"$set": doc}, upsert=True) for doc in data]

        if operations:
            db_store.bulk_write(operations)

    def delete_all(self):
        """Delete all documents in the NoSQL database."""
        self.maybe_create_db()
        self.collection.delete_many({})
        if self.collection.count_documents({}) == 0:
            self.log("All documents deleted successfully.")

    def query_check_db(self, limit: int = 100) -> DataFrame:
        """Check if the database is available and the collection exists."""
        self.maybe_create_db()

        top = self.collection.find().limit(limit)
        return DataFrame(list(top))

    def __convert_data_to_list(self):
        """Convert Data or DataFrame to a list of dictionaries."""
        self.log(f"Converting query data to list: {self.query}")
        list_idx = [doc.data["faiss_id"] for doc in self.query]
        self.log(f"Converted: {self.query}")
        self.query = list_idx

    def __execute_query(self) -> list[dict[str, Any]]:
        self.maybe_create_db()
        self.log(f"Running query: {self.query}")

        try:
            # cursor: Result[Any] = self.db.run(self.query, fetch="cursor")
            result = self.collection.find({self.column_id: {"$in": self.query}})
            return list(result)
        except SQLAlchemyError as e:
            msg = f"An error occurred while running the SQL Query: {e}"
            self.log(msg)
            raise ValueError(msg) from e

    def run_sql_query(self) -> DataFrame:
        """Run the SQL query against the NoSQL database."""
        if self.mode == "Index":
            self.__ingest(index=True)
            self.log("Ingested data into the NoSQL database.")
            return DataFrame([])
        self.__convert_data_to_list()
        result = self.__execute_query()
        self.log(f"result: {result}")
        df_result = DataFrame(result)
        self.status = df_result
        if df_result:
            return df_result
        self.log("No results found for the query.")
        return DataFrame([])
