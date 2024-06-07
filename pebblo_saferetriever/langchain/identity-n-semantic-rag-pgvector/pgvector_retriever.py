import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from google_auth import get_authorized_identities
from utils import format_text

load_dotenv()

PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
COLLECTION_NAME = "identity-enabled-rag"


class SafeRetrieverSemanticRAG:
    def __init__(self, folder_id: str):
        self.loader_app_name = "pebblo-identity-n-semantic-loader"
        self.retrieval_app_name = "pebblo-identity-n-semantic-retriever"
        self.embeddings = OpenAIEmbeddings()

        # Load documents into VectorDB
        print("Hydrating Vector DB ...")
        self.vectordb = self.init_vectordb()
        print("Finished hydrating Vector DB ...\n")

    def init_vectordb(self):
        """
        Initialize VectorDB
        """
        vectordb = PGVector(
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=PG_CONNECTION_STRING,
            use_jsonb=True,
        )
        print("VectorDB initialized")
        return vectordb

    def ask(
        self,
        question: str,
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
        # retriever = self.vectordb.as_retriever(
        #     search_kwargs={"filter": {"authorized_identities": {"$eq": auth_identifiers}}}
        # )
        retriever = self.vectordb.as_retriever(
            search_kwargs={
                "filter": {"authorized_identities": {"$eq": auth_identifiers}}
            }
        )
        docs: List[Document] = retriever.invoke(question)
        for doc in docs:
            print(
                f"\ncontent: {doc.page_content}\n"
                f"\ttitle: {doc.metadata.get('title')}\n"
                f"\tauthorized_identities: {doc.metadata.get('authorized_identities')}\n"
                f"\ttopics: {doc.metadata.get('pebblo_semantic_topics')}\n"
                f"\tentities: {doc.metadata.get('pebblo_semantic_entities')}"
            )
        print(f"total docs: {len(docs)}")
        return docs


if __name__ == "__main__":
    rag_app = SafeRetrieverSemanticRAG(folder_id="input_folder_id")

    print("Please enter end user details below")

    # prompt = "Share performance details for Mark Johnson"
    prompt = "Share Mark Johnson details"
    # prompt = "Share performance details for John Smith"
    end_user_email_address = "demo-user-hr@daxa.ai"
    topic_to_deny = []
    entity_to_deny = []

    auth_identifiers = get_authorized_identities(
        admin_user_email_address="sridhar@clouddefense.io",
        credentials_file_path="credentials/service-account.json",
        user_email=end_user_email_address,
    )

    print(
        f"User: {end_user_email_address}.\n"
        f"Topics to deny: {topic_to_deny}\n"
        f"Entities to deny: {entity_to_deny}\n"
        f"Auth Identifiers: {auth_identifiers}\n"
        f"Query: {format_text(prompt)}"
    )

    response = rag_app.ask(
        prompt,
        auth_identifiers,
        topic_to_deny,
        entity_to_deny,
    )
