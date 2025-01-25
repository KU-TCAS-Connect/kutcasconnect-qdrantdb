import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


load_dotenv()

client = QdrantClient("http://localhost:6333")
collection_name = "admission_records"

def generate_openai_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def search_similar_vectors(query_text, top_k=5):
    query_embedding = generate_openai_embedding(query_text)
    if not query_embedding:
        print("Failed to generate query embedding.")
        return

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    for result in search_results:
        print(f"Found ID: {result.id}, Score: {result.score}, Metadata: {result.payload}")


query = "เกณฑ์ของภาควิศวกรรมซอฟต์แวร์และความรู้ รอบ1/1 นานาชาติและภาษาอังกฤษ"
search_result = client.search(
    collection_name=collection_name,
    query_vector=generate_openai_embedding(query),
    query_filter=Filter(
        must=[
            FieldCondition(key="admission_round", match=MatchValue(value="1/1")),
            FieldCondition(key="admission_program", match=MatchValue(value="นานาชาติและภาษาอังกฤษ")),
        ]
    ),
    with_payload=True,
    limit=5,
)

for result in search_result:
    print(f"Found ID: {result.id}, Score: {result.score}, Metadata: {result.payload}")