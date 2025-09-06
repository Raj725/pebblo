import logging
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from langchain_community.document_loaders.slack_api import SlackAPILoader
from mpmath.calculus.extrapolation import limit

load_dotenv()

import os
from langchain_community.chains import PebbloRetrievalQA
from langchain_community.chains.pebblo_retrieval.models import (
    AuthContext,
    ChainInput,
    SemanticContext,
)
from langchain_community.document_loaders.pebblo import PebbloSafeLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain_postgres import PGVector
from utils import get_input_as_list, format_text
from langchain_community.utilities.slack import     DEFAULT_MESSAGE_LIMIT

PEBBLO_API_KEY = os.getenv("PEBBLO_API_KEY")
PEBBLO_CLOUD_URL = os.getenv("PEBBLO_CLOUD_URL")
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PebbloSafeRAG:
    """
    Sample app to demonstrate the usage of PebbloSafeLoader, and PebbloRetrievalQA for Identity & Semantic enforcement
    using SharePointLoader and PostgreSQL VectorDB
    """

    def __init__(self, channel_name: str, message_limit:int, collection_name: str):
        self.loader_app_name = "pebblo-safe-loader-slack-1"
        self.retrieval_app_name = "pebblo-safe-retriever-slack-1"
        self.collection_name = collection_name
        self.channel_name = channel_name

        print(120 * "-")
        # Load documents
        print("Loading RAG documents ...")
        token_path = Path.home() / ".credentials" / "o365_token.txt"
        self.loader = PebbloSafeLoader(
            SlackAPILoader(
                channel_name=self.channel_name,
                message_limit=message_limit,
                load_auth=True,
            ),
            name=self.loader_app_name,  # App name (Mandatory)
            owner="Joe Smith",  # Owner (Optional)
            description="Identity & Semantic enabled SafeLoader and SafeRetrival app using Pebblo",
            # Description (Optional)
            load_semantic=True,
            api_key=PEBBLO_API_KEY,
        )
        self.documents = self.loader.load()
        unique_identities = set()
        unique_topics = set()
        unique_entities = set()

        for doc in self.documents:
            if doc.metadata.get("authorized_identities"):
                unique_identities.update(doc.metadata.get("authorized_identities"))
            if doc.metadata.get("pebblo_semantic_topics"):
                unique_topics.update(doc.metadata.get("pebblo_semantic_topics"))
            if doc.metadata.get("pebblo_semantic_entities"):
                unique_entities.update(doc.metadata.get("pebblo_semantic_entities"))

        print(f"Loaded {len(self.documents)} documents with the following metadata:")
        print(f"Authorized Identities: {list(unique_identities)}")
        print(f"Semantic Topics: {list(unique_topics)}")
        print(f"Semantic Entities: {list(unique_entities)}")
        print(120 * "-")

        # Load documents into VectorDB
        print("Hydrating Vector DB ...")
        self.vectordb = self.init_vector_db()
        print("Finished hydrating Vector DB ...\n")

        # Prepare LLM
        self.llm = OpenAI()
        print("Initializing PebbloRetrievalQA ...")
        self.retrieval_chain = self.init_retrieval_chain()
        print("Finished initializing PebbloRetrievalQA ...")
        print(120 * "-")

    def init_retrieval_chain(self):
        """
        Initialize PebbloRetrievalQA chain
        """
        return PebbloRetrievalQA.from_chain_type(
            llm=self.llm,
            app_name=self.retrieval_app_name,
            owner="Joe Smith",
            description="Identity enabled filtering using PebbloSafeLoader, and PebbloRetrievalQA",
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            verbose=True,
        )

    def init_vector_db(self):
        """
        Initialize PostgreSQL VectorDB from documents
        """
        embeddings = OpenAIEmbeddings()
        vectordb = PGVector.from_documents(
            embedding=embeddings,
            documents=self.documents,
            collection_name=self.collection_name,
            connection=PG_CONNECTION_STRING,
            pre_delete_collection=True,
            use_jsonb=True,
        )
        print(f"Added {len(self.documents)} documents to PostgreSQL ...\n")
        return vectordb

    def ask(
            self,
            question: str,
            user_email: str,
            auth_identifiers: list,
            topics_to_deny: Optional[List[str]] = None,
            entities_to_deny: Optional[List[str]] = None,
    ):
        """
        Ask a question with identity and semantic context
        """
        auth_context = None
        if auth_identifiers:
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
        # Print chain input in formatted json
        print(f"\nchain_input: {chain_input.model_dump_json(indent=4)}")
        return self.retrieval_chain.invoke(chain_input.dict())


if __name__ == "__main__":
    input_collection_name = "identity-enabled-rag-slack-1"

    print("Please enter the slack app details below:")

    channel_name = input(f"Slack channel name(leave empty for all): ")
    # Strip the channel name
    channel_name = channel_name.strip() if channel_name else None

    # Number of messages to load
    def_num_messages = DEFAULT_MESSAGE_LIMIT
    num_messages = input(f"Number of messages to load(default={def_num_messages}): ") or def_num_messages

    rag_app = PebbloSafeRAG(
        channel_name=channel_name,
        message_limit=int(num_messages),
        collection_name=input_collection_name,
    )

    # Ask questions
    print(120 * "-")
    print("Welcome to PebbloSafeRAG...\n")

    while True:
        print("Please enter end user details to ask a question:")
        end_user_email_address = input("User email address: ")

        # def_topics = None  # ["employee-agreement"]
        # topic_to_deny = (
        #         get_input_as_list(
        #             f"Enter topics to deny (Optional, comma separated, no quotes needed): "
        #         )
        #         or def_topics
        # )
        topic_to_deny = None

        # def_entities = None
        # entity_to_deny = (
        #         get_input_as_list(
        #             f"Enter entities to deny (Optional, comma separated, no quotes needed): "
        #         )
        #         or def_entities
        # )
        entity_to_deny = None


        # prompt = input("Please provide the prompt : ")
        prompt = input("Ask a question: ")

        authorized_identities = [end_user_email_address]

        response = rag_app.ask(
            prompt,
            end_user_email_address,
            authorized_identities,
            topic_to_deny,
            entity_to_deny,
        )
        print(f"Result: {format_text(response['result'])}")
        print(120 * "-")

        try:
            continue_or_exist = int(input("\nType 1 to continue and 0 to exit : "))
        except ValueError:
            print("Please provide valid input")
            continue

        if not continue_or_exist:
            exit(0)

        print("\n")
