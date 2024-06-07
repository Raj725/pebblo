# pip install psycopg2
import json

import psycopg2


# PG_CONNECTION_STRING = "postgresql://postgres:postgress@localhost:32207/postgres"


def print_result(_results, _filter_criteria=None):
    """
    Get documents from the vectorstore based on the query string and filter criteria.
    """
    # print(120 * "-")
    try:
        if _filter_criteria:
            print("Filter Criteria:", _filter_criteria)
        print(f"Results: {len(_results)} documents found....\n")
        if len(_results) == 0:
            print("No matching documents found....")
        else:
            for doc in _results:
                print(doc)
    except Exception as e:
        print(f"Error: {e}")
    print(120 * "-" + "\n")


def run_query(query):
    # Connect to your postgres DB
    conn = psycopg2.connect(
        dbname="postgres",  # "your_dbname
        user="postgres",  # "your_username
        password="postgress",  # "your_password
        host="localhost",  # "your_host"
        port="32207",  # "your_port"
    )

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Execute the query
    cur.execute(query)

    # Fetch the results
    results = cur.fetchall()

    # Close the cursor and the connection
    cur.close()
    conn.close()

    return results


if __name__ == "__main__":
    # query_prefix = """
    #     --langchain_pg_embedding.collection_id AS langchain_pg_embedding_collection_id,
    #     --langchain_pg_embedding.embedding AS langchain_pg_embedding_embedding,
    #     --langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata,
    #     --langchain_pg_embedding.custom_id AS langchain_pg_embedding_custom_id,
    #     --langchain_pg_embedding.uuid AS langchain_pg_embedding_uuid
    # """

    query_prefix = """
    SELECT 
        langchain_pg_embedding.document AS langchain_pg_embedding_document, 
        langchain_pg_embedding.cmetadata AS langchain_pg_embedding_cmetadata 
    FROM langchain_pg_embedding 
    JOIN langchain_pg_collection 
        ON langchain_pg_embedding.collection_id = langchain_pg_collection.uuid
    WHERE 
        langchain_pg_collection.name = 'identity-enabled-rag-3'        
    """

    # 1. Get all documents having the topic "food"
    print("1. Get all documents having the topic 'food'")
    filter_value = ["food"]
    filter_value_json = json.dumps(filter_value)
    and_filter = f"""AND jsonb_path_match(langchain_pg_embedding.cmetadata, '$.topics == $value', '{{"value": {filter_value_json}}}')"""
    query = query_prefix + " " + and_filter
    results = run_query(query)
    print_result(results, filter_value)

    # 2. Get all documents having the topic "food" or "animals"
    print("2. Get all documents having the topic 'food' or 'animals'")
    filter_value = ["food", "animals"]
    filter_value_json = json.dumps(filter_value)
    or_filter = f"""AND jsonb_path_match(langchain_pg_embedding.cmetadata, '$.topics == $value', '{{"value": {filter_value_json}}}')"""
    query = query_prefix + " " + or_filter
    results = run_query(query)
    print_result(results, filter_value)

    # 3. Get all documents having no "food" topic
    print("3. Get all documents having no 'food' topic")
    filter_value = ["food"]
    filter_value_json = json.dumps(filter_value)
    not_filter = f"""AND NOT jsonb_path_match(langchain_pg_embedding.cmetadata, '$.topics == $value', '{{"value": {filter_value_json}}}')"""
    query = query_prefix + " " + not_filter
    results = run_query(query)
    print_result(results, filter_value)

    # 4. Get all documents having no "food" or "animals"
    print("4. Get all documents having no 'food' or 'animals'")
    filter_value = ["food", "animals"]
    filter_value_json = json.dumps(filter_value)
    not_filter = f"""AND NOT jsonb_path_match(langchain_pg_embedding.cmetadata, '$.topics == $value', '{{"value": {filter_value_json}}}')"""
    query = query_prefix + " " + not_filter
    results = run_query(query)
    print_result(results, filter_value)
