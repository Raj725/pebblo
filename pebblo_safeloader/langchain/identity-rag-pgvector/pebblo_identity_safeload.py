import os
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredFileIOLoader
from langchain_community.document_loaders.pebblo import PebbloSafeLoader
from langchain_google_community import GoogleDriveLoader
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores.pgvector import PGVector

from langchain_postgres import PGVector

load_dotenv()

PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
COLLECTION_NAME = "identity-enabled-rag"


class IdentityBasedDataLoader:
    def __init__(self, folder_id: str):
        self.app_name = "acme-corp-rag-1"
        self.folder_id = folder_id
        self.embeddings = OpenAIEmbeddings()

    def load_documents(self):
        print("\nLoading RAG documents ...")
        loader = PebbloSafeLoader(
            GoogleDriveLoader(
                folder_id=self.folder_id,
                # credentials_path="credentials/credentials.json",
                token_path="./google_token.json",
                recursive=True,
                file_loader_cls=UnstructuredFileIOLoader,
                file_loader_kwargs={"mode": "elements"},
                load_auth=True,
            ),
            name=self.app_name,  # App name (Mandatory)
            owner="Joe Smith",  # Owner (Optional)
            description="Identity enabled SafeLoader and SafeRetrival app using Pebblo",  # Description (Optional)
            load_semantic=True,
        )
        documents = loader.load()
        unique_identities = set()
        for doc in documents:
            unique_identities.update(doc.metadata.get("authorized_identities"))

        print(f"Authorized Identities: {list(unique_identities)}")
        print(f"Loaded {len(documents)} documents ...\n")
        return documents

    def add_docs_to_vectordb(self, documents: List[Document]):
        """
        Load documents into PostgreSQL
        """
        print("\nAdding documents to PostgreSQL ...")
        # vectordb = PGVector.from_documents(
        #     embedding=self.embeddings,
        #     documents=documents,
        #     collection_name=COLLECTION_NAME,
        #     connection_string=PG_CONNECTION_STRING,
        #     pre_delete_collection=True,
        #     use_jsonb=True,
        # )
        vectordb = PGVector.from_documents(
            embedding=self.embeddings,
            documents=documents,
            collection_name=COLLECTION_NAME,
            # connection_string=PG_CONNECTION_STRING,
            connection=PG_CONNECTION_STRING,
            pre_delete_collection=True,
            use_jsonb=True,
        )
        print(f"Added {len(documents)} documents to PostgreSQL ...\n")
        return vectordb


if __name__ == "__main__":
    print("Loading documents to PostgreSQL ...")
    def_folder_id = "15CyFIWOPJOR5BxDID7G6tUisfHU1szrg"

    qloader = IdentityBasedDataLoader(def_folder_id)

    result_documents = qloader.load_documents()

    vectordb_obj = qloader.add_docs_to_vectordb(result_documents)
