import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.chains import PebbloRetrievalQA
from langchain_community.chains.pebblo_retrieval.models import (
    SemanticContext,
    ChainInput,
    AuthContext,
)
from langchain_community.document_loaders import (
    PebbloSafeLoader,
    UnstructuredFileIOLoader,
)
from langchain_google_community import GoogleDriveLoader
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores.pgvector import PGVector

from langchain_postgres import PGVector
from langchain_openai.llms import OpenAI
from google_auth import get_authorized_identities
from utils import format_text, get_input_as_list

load_dotenv()

PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
COLLECTION_NAME = "identity-enabled-rag"


class SafeRetrieverSemanticRAG:
    def __init__(self, folder_id: str):
        self.loader_app_name = "pebblo-identity-n-semantic-loader"
        self.retrieval_app_name = "pebblo-identity-n-semantic-retriever"
        self.folder_id = folder_id
        self.embeddings = OpenAIEmbeddings()

        self.documents = self.load_documents()
        # Load documents into VectorDB
        print("Hydrating Vector DB ...")
        self.vectordb = self.add_docs_to_vectordb(self.documents)
        print("Finished hydrating Vector DB ...\n")

        # Prepare LLM
        self.llm = OpenAI()
        print("Initializing PebbloRetrievalQA ...")
        self.retrieval_chain = self.init_retrieval_chain()

    def load_documents(self):
        """
        Load documents with Identity and Semantic metadata using PebbloSafeLoader
        """
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
            name=self.loader_app_name,  # App name (Mandatory)
            owner="Joe Smith",  # Owner (Optional)
            description="Identity enabled SafeLoader and SafeRetrival app using Pebblo",  # Description (Optional)
            load_semantic=True,
        )
        documents = loader.load()
        unique_identities = set()
        unique_topics = set()
        unique_entities = set()

        for doc in documents:
            if doc.metadata.get("authorized_identities"):
                unique_identities.update(doc.metadata.get("authorized_identities"))
            if doc.metadata.get("pebblo_semantic_topics"):
                unique_topics.update(doc.metadata.get("pebblo_semantic_topics"))
            if doc.metadata.get("pebblo_semantic_entities"):
                unique_entities.update(doc.metadata.get("pebblo_semantic_entities"))

        print(f"Authorized Identities: {list(unique_identities)}")
        print(f"Semantic Topics: {list(unique_topics)}")
        print(f"Semantic Entities: {list(unique_entities)}")
        print(f"Loaded {len(documents)} documents ...\n")
        return documents

    def add_docs_to_vectordb(self, documents: List[Document]):
        """
        Load documents into PostgreSQL
        """
        print("\nAdding documents to PostgreSQL ...")

        # vectorstore = PGVector(
        #     # embedding_function=embeddings,
        #     # connection_string=PG_CONNECTION_STRING,
        #     embeddings=embeddings,
        #     connection=PG_CONNECTION_STRING,
        #     collection_name=COLLECTION_NAME,
        #     use_jsonb=True,
        # )

        # Working, with langchain-community
        # vectordb = PGVector.from_documents(
        #     embedding=self.embeddings,
        #     documents=documents,
        #     collection_name=COLLECTION_NAME,
        #     # connection_string=PG_CONNECTION_STRING,
        #     connection=PG_CONNECTION_STRING,
        #     pre_delete_collection=True,
        #     use_jsonb=True,
        # )

        # Working, with langchain-postgres
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

    def init_retrieval_chain(self):
        """
        Initialize PebbloRetrievalQA chain
        """
        return PebbloRetrievalQA.from_chain_type(
            llm=self.llm,
            # app_name=self.retrieval_app_name,
            # owner="Joe Smith",
            # description="Identity and Semantic filtering using PebbloSafeLoader, and PebbloRetrievalQA",
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            verbose=True,
        )

    def ask(
        self,
        question: str,
        user_email: str,
        auth_identifiers: List[str],
        topics_to_deny: Optional[List[str]] = None,
        entities_to_deny: Optional[List[str]] = None,
    ):
        """
        Ask a question to the retrieval chain

        Args:
            question (str): The question to ask
            user_email (str): The user email
            auth_identifiers (List[str]): The authorized identifiers
            topics_to_deny (Optional[List[str]]): The topics to deny
            entities_to_deny (Optional[List[str]]): The entities to deny
        """

        retrieval_chain = PebbloRetrievalQA.from_chain_type(
            llm=self.llm,
            # app_name=self.retrieval_app_name,
            # owner="Joe Smith",
            # description="Identity and Semantic filtering using PebbloSafeLoader, and PebbloRetrievalQA",
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            verbose=True,
        )

        auth_context = {
            "user_id": user_email,
            "user_auth": auth_identifiers,
        }
        auth_context = AuthContext(**auth_context)
        semantic_context = dict()
        if topics_to_deny:
            semantic_context["pebblo_semantic_topics"] = {"deny": topics_to_deny}
        if entities_to_deny:
            semantic_context["pebblo_semantic_entities"] = {"deny": entities_to_deny}

        semantic_context = (
            SemanticContext(**semantic_context) if semantic_context else None
        )

        chain_input = ChainInput(
            query=question, auth_context=auth_context, semantic_context=semantic_context
        )

        return retrieval_chain.invoke(chain_input.dict())


if __name__ == "__main__":
    service_acc_def = "credentials/service-account.json"
    in_folder_id_def = "15CyFIWOPJOR5BxDID7G6tUisfHU1szrg"
    ing_user_email_def = "sridhar@clouddefense.io"

    print("Please enter ingestion user details for loading data...")
    ingestion_user_email_address = (
        input(f"email address ({ing_user_email_def}): ") or ing_user_email_def
    )
    ingestion_user_service_account_path = (
        input(f"service-account.json path ({service_acc_def}): ") or service_acc_def
    )
    input_folder_id = input(f"Folder id ({in_folder_id_def}): ") or in_folder_id_def

    rag_app = SafeRetrieverSemanticRAG(folder_id=input_folder_id)

    while True:
        print("Please enter end user details below")
        end_user_email_address = (
            input("User email address : ") or "demo-user-hr@daxa.ai"
        )

        auth_identifiers = get_authorized_identities(
            admin_user_email_address=ingestion_user_email_address,
            credentials_file_path=ingestion_user_service_account_path,
            user_email=end_user_email_address,
        )

        print(
            "Please enter semantic filters below...\n"
            "(Leave these fields empty if you do not wish to enforce any semantic filters)"
        )
        topic_to_deny = get_input_as_list(
            "Topics to deny, comma separated (Optional): "
        )
        entity_to_deny = get_input_as_list(
            "Entities to deny, comma separated (Optional): "
        )

        prompt = input("Please provide the prompt: ")

        print(
            f"User: {end_user_email_address}.\n"
            f"Topics to deny: {topic_to_deny}\n"
            f"Entities to deny: {entity_to_deny}\n"
            f"Auth Identifiers: {auth_identifiers}\n"
            f"Query: {format_text(prompt)}"
        )

        response = rag_app.ask(
            prompt,
            end_user_email_address,
            auth_identifiers,
            topic_to_deny,
            entity_to_deny,
        )

        print(f"Response:\n" f"{format_text(response['result'])}")

        try:
            continue_or_exist = int(
                input("\n\nType 1 to continue and 0 to exit (1): ") or 1
            )
        except ValueError:
            print("Please provide valid input")
            continue

        if not continue_or_exist:
            exit(0)

        print("\n\n")
