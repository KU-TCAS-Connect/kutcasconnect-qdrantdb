import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

client = QdrantClient("http://localhost:6333")
collection_name = "ku_tcas_document_bge"

# Initialize the BGEM3 model
bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def compute_sparse_vector(text):
    sentences_1 = [text]  # Use the content of the row for encoding
    output_1 = bge_model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    # Extract the lexical weights (this is your sparse representation)
    lexical_weights = output_1['lexical_weights'][0]

    # Convert the lexical weights into a dictionary (index: weight)
    sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

    indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
    values = list(sparse_vector_dict.values())  # Values of the sparse vector
    native_floats = [float(x) for x in values]
    new_dict = dict(zip(indices, native_floats))
    return indices, native_floats

def generate_bge_embedding(text):
    try:
        # Generate dense embedding using BGE model
        sentences_1 = [text]  # The content you want to encode
        output_1 = bge_model.encode(sentences_1, return_dense=True, return_sparse=False, return_colbert_vecs=False)

        # Extract the dense vector (embedding)
        dense_vector = output_1['dense_vecs'][0]

        return dense_vector
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def search_similar_vectors(query_text, top_k=5):
    query_embedding = generate_bge_embedding(query_text)  # Use BGE model for dense vector
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


query = "อยากทราบกำหนดการการประกาศผลผู้ผ่านการคัดเลือก"
query_indices, query_values = compute_sparse_vector(query)

search_result = client.query_points(
    collection_name=collection_name,
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=query_indices, values=query_values),
            using="keywords",
            limit=5,
        ),
        models.Prefetch(
            query=generate_bge_embedding(query),  # <-- dense vector using BGE model
            using="",
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
)

# Print the search results
for result in search_result.points:
    print(f"Score: {result.score}")
    print(f"""{result.payload["admission_program"]}\n{result.payload["contents"]}\n{result.payload["reference"]}""")
    print("---------------------------------")
