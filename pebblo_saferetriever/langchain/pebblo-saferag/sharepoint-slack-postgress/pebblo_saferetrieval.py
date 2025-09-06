import os
from typing import Optional, List

from dotenv import load_dotenv

load_dotenv()
from utils import get_input_as_list, format_text

from langchain_openai import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain_postgres import PGVector

from langchain_community.chains import PebbloRetrievalQA
from langchain_community.chains.pebblo_retrieval.models import (
    AuthContext,
    ChainInput,
    SemanticContext,
)

PEBBLO_API_KEY = os.getenv("PEBBLO_API_KEY")
PEBBLO_CLOUD_URL = os.getenv("PEBBLO_CLOUD_URL")
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

print("PG_CONNECTION_STRING", PG_CONNECTION_STRING)


class PebbloSemanticIdentityRAG:
    """
    Pebblo Semantic Identity RAG using PGVector
    """

    def __init__(
        self, collection_name: str, app_name: str = "acme-retrieval-pebblo-app"
    ):
        self.app_name = app_name
        self.pg_collection_name = collection_name
        # self.llm = Ollama(model="phi3")
        self.llm = OpenAI()
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = self.init_vector_db()
        self.retrieval_chain = self.init_retrieval_chain()

    def init_vector_db(self):
        """
        Load Vector DB from file
        """
        vectordb = PGVector(
            embeddings=self.embeddings,
            connection=PG_CONNECTION_STRING,
            collection_name=self.pg_collection_name,
            use_jsonb=True,
        )
        return vectordb

    def init_retrieval_chain(self):
        """
        Initialize PebbloRetrievalQA chain
        """
        retriever_num_docs_returned = 3 # top_k
        retriever_num_docs_used = 100    # fetck_k
        retriever=self.vectordb.as_retriever(search_kwargs={"k": retriever_num_docs_returned, "fetch_k": retriever_num_docs_used})

        return PebbloRetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            app_name=self.app_name,  # App name (Mandatory)
            owner="Joe Smith",
            description="Multi-user Knowledge-Base RAG application",
            verbose=True,
            # api_key=os.environ.get("PEBBLO_API_KEY"),
        )

    def ask(
        self,
        question: str,
        user_email: str,
        auth_identifiers: list,
        topics_to_deny: Optional[List[str]] = None,
        entities_to_deny: Optional[List[str]] = None,
    ):
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
        print(f"chain_input: {chain_input.dict()} ...\n")
        return self.retrieval_chain.invoke(chain_input.dict())


if __name__ == "__main__":
    rag_app = PebbloSemanticIdentityRAG(collection_name="identity-enabled-rag-slack-1")

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