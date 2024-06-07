from dotenv import load_dotenv

# from langchain_community.vectorstores.pgvector import PGVector

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

# PG_CONNECTION_STRING = "postgresql://postgres:postgress@localhost:32207/postgres"
# PG_CONNECTION_STRING = "postgresql+psycopg2://tsdbadmin:kf4r3qhh79x5chom@d0ckqst8oy.knf482ps4y.tsdb.cloud.timescale.com:32207/tsdb?sslmode=require"
PG_CONNECTION_STRING = "postgresql://postgres:postgress@localhost:32207/rag"
COLLECTION_NAME = "identity-enabled-rag-5"

embeddings = OpenAIEmbeddings()
vectorstore = PGVector(
    # embedding_function=embeddings,
    # connection_string=PG_CONNECTION_STRING,
    embeddings=embeddings,
    connection=PG_CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    use_jsonb=True,
)


def get_documents(query_string: str, filter_criteria: dict):
    """
    Get documents from the vectorstore based on the query string and filter criteria.
    """
    print(120 * "-")
    try:
        res = vectorstore.similarity_search(
            query_string,
            k=10,
            filter=filter_criteria,
        )
        print("Query String:", query_string)
        print("Filter Criteria:", filter_criteria)
        print(f"Results: {len(res)} documents found....")
        if len(res) == 0:
            print("No matching documents found....")
        else:
            for doc in res:
                print(doc)
    except Exception as e:
        print(f"Error: {e}")
    print(120 * "-")


docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={
            "id": 1,
            "location": "pond",
            "topic": "animals",
            "topics": ["nature", "animals"],
        },
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={
            "id": 2,
            "location": "pond",
            "topic": "animals",
            "topics": ["nature", "animals"],
        },
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={
            "id": 3,
            "location": "market",
            "topic": "food",
            "topics": ["market", "food"],
        },
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={
            "id": 4,
            "location": "market",
            "topic": "food",
            "topics": ["market", "food"],
        },
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={
            "id": 5,
            "location": "museum",
            "topic": "art",
            "topics": ["museum", "art"],
        },
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={
            "id": 6,
            "location": "museum",
            "topic": "art",
            "topics": ["museum", "art"],
        },
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={
            "id": 7,
            "location": "Main Street",
            "topic": "food",
            "topics": ["Main Street", "food"],
        },
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={
            "id": 8,
            "location": "library",
            "topic": "reading",
            "topics": ["library", "reading"],
        },
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={
            "id": 9,
            "location": "library",
            "topic": "reading",
            "topics": ["library", "reading"],
        },
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={
            "id": 10,
            "location": "community center",
            "topic": "classes",
            "topics": ["cooking", "classes"],
        },
    ),
]

# vectorstore.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

# # Correctly returns list of documents with "food" in topic
# filters = {"topic": {"$in": ["food"]}}
# get_documents("sell", filters)
#
# Does not return any documents as $in works only with string fields
filters = {"topics": {"$in": ["food"]}}
get_documents("sell", filters)

# Correctly returns list of documents with "food" or "animals" in topics
filters = {"topics": {"$in": ["food", "animals"]}}
get_documents("sell", filters)

# Correctly returns list of documents with "food" in topics
filters = {"topics": {"$eq": ["food"]}}
get_documents("sell", filters)

# Correctly returns list of documents with "food" or "animals" in topics
filters = {"topics": {"$eq": ["food", "animals"]}}
get_documents("sell", filters)

# # Correctly returns no documents as none have "ecom" and "business" in topics
# filters = {"topics": {"$eq": ["ecom", "business"]}}
# get_documents("sell", filters)
#
# # (Incorrect) Returns all documents
# filters = {"topics": {"$ne": ["food", "animals"]}}
# get_documents("library", filters)

# Correctly returns list of documents with no "food", "cooking" and "animals" in topics
filters = {
    "$not": [
        {"topics": {"$eq": ["food", "cooking"]}},
        {"topics": {"$eq": ["animals", "banned"]}},
    ]
}
get_documents("sell", filters)

# # Correctly returns list of documents with no "food" in topics
# filters = {
#     "$and": [
#         {"topics": {"$eq": ["museum"]}},
#         {
#             "$not": [
#                 {"topics": {"$eq": ["food", "cooking"]}},
#                 {"topics": {"$eq": ["animals", "banned"]}},
#             ]
#         },
#     ]
# }
# get_documents("sell", filters)


# Throws error (AttributeError: Neither 'BinaryExpression' object nor 'Comparator' object has an attribute 'nin_')
filters = {"topics": {"$nin": ["food", "animals"]}}
get_documents("sell", filters)

# Test complex filter
filters = {
    "$and": [
        {"topics": {"$eq": ["museum"]}},
        {
            "$or": [
                {
                    "$not": [
                        {"topics": {"$eq": ["food", "cooking"]}},
                        {"topics": {"$eq": ["animals", "banned"]}},
                    ]
                },
                {
                    "$not": [{"topics": {"$eq": ["food", "cooking"]}}],
                },
            ]
        },
    ],
}
get_documents("sell", filters)
