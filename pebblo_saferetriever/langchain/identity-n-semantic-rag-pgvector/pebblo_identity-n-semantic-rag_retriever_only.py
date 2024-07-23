import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.chains import PebbloRetrievalQA
from langchain_community.chains.pebblo_retrieval.models import (
    SemanticContext,
    ChainInput,
    AuthContext,
)
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
    def __init__(self):
        self.loader_app_name = "pebblo-identity-n-semantic-loader"
        self.retrieval_app_name = "pebblo-identity-n-semantic-retriever"
        self.embeddings = OpenAIEmbeddings()

        # Load documents into VectorDB
        self.vectordb = self.init_vectordb()

        # Prepare LLM
        self.llm = OpenAI()
        print("Initializing PebbloRetrievalQA ...")
        self.retrieval_chain = self.init_retrieval_chain()

    def init_vectordb(self):
        """
        Initialize VectorDB
        """
        # Langchain-community
        # vectordb = PGVector(
        #     embedding_function=self.embeddings,
        #     collection_name=COLLECTION_NAME,
        #     connection_string=PG_CONNECTION_STRING,
        #     use_jsonb=True,
        # )

        # Langchain-postgres
        vectordb = PGVector(
            embeddings=self.embeddings,
            connection=PG_CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
        )
        print("VectorDB initialized")
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
        # retrieval_chain = PebbloRetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     # app_name=self.retrieval_app_name,
        #     # owner="Joe Smith",
        #     # description="Identity and Semantic filtering using PebbloSafeLoader, and PebbloRetrievalQA",
        #     chain_type="stuff",
        #     retriever=self.vectordb.as_retriever(
        #         # search_kwargs={
        #         #     "filter": {"authorized_identities": {"$eq": auth_identifiers}}
        #         # }
        #     ),
        #     verbose=True,
        # )

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
        print(f"chain_input: {chain_input.dict()} ...\n")
        return self.retrieval_chain.invoke(chain_input.dict())


if __name__ == "__main__":
    service_acc_def = "credentials/service-account.json"
    ing_user_email_def = "sridhar@clouddefense.io"

    print("Please enter ingestion user details for loading data...")
    ingestion_user_email_address = ing_user_email_def
    ingestion_user_service_account_path = service_acc_def

    rag_app = SafeRetrieverSemanticRAG()

    while True:
        print("Please enter end user details below")
        end_user_email_address = (
            input("User email address (demo-user-hr@daxa.ai): ")
            or "demo-user-hr@daxa.ai"
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
