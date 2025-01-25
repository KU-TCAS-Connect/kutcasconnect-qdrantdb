import uuid
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams
import ast
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")

client = QdrantClient("http://localhost:6333")
collection_name = "admission_records_test"

# Initialize the BGEM3 model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # Use FP16 for faster computation

# Create a collection (if not already created)
def create_collection():
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,  # Size for the dense vector (for example)
            distance=Distance.COSINE  # Choose from COSINE, EUCLIDEAN, DOT
        ),
        sparse_vectors_config={
            "keywords": models.SparseVectorParams(  # Field name for sparse vectors
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"Collection '{collection_name}' created successfully.")

def uuid_from_time(timestamp):
    return uuid.uuid5(uuid.NAMESPACE_DNS, timestamp.isoformat())

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def process_and_insert_data(df):
    points = []
    for _, row in df.iterrows():
        admission_round = row.get('round', 'N/A')
        admission_program = row.get('program_type', 'N/A')
        content = row.get('content', 'N/A')

        # Use BGEM3 to generate dense and sparse vectors
        sentences_1 = [content]  # Use the content of the row for encoding

        output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        # Extract the lexical weights (this is your sparse representation)
        lexical_weights = output_1['lexical_weights'][0]

        # Convert the lexical weights into a dictionary (index: weight)
        sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

        try:
            embedding = output_1['dense'][0]  # Dense vector from BGEM3
            if not isinstance(embedding, list):
                raise ValueError("Embedding is not a valid list")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing embedding for row {row['สาขาวิชา']}: {e}")
            continue 

        data = {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "major": row['สาขาวิชา'],
                "admission_round": admission_round,
                "admission_program": admission_program,
                "reference": row['แหล่งที่มา'],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
            "lexical_weights": sparse_vector_dict,  # Add lexical_weights here
        }

        # Prepare the sparse vector: list of indices and values for sparse vector
        indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
        values = list(sparse_vector_dict.values())  # Values of the sparse vector

        # Create the point with the embedding and sparse vector
        point = PointStruct(
            id=data["id"],
            vector={
                "": data["embedding"],  # Dense vector
                "keywords": models.SparseVector(  # Sparse vector with "keywords"
                    indices=indices,  # List of indices
                    values=values  # List of values
                ),
            },
            payload={
                "major": data["metadata"]["major"],
                "admission_round": data["metadata"]["admission_round"],
                "admission_program": data["metadata"]["admission_program"],
                "reference": data["metadata"]["reference"],
                "created_at": data["metadata"]["created_at"],
                "contents": data["contents"],
                "lexical_weights": sparse_vector_dict,  # Store sparse vector in payload
            },
        )

        print(data)
        print(point)
        print("-------------------------------------------------------")
        points.append(point)

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Inserted {len(points)} records into Qdrant.")
    else:
        print("No records were inserted due to embedding failures.")

if __name__ == "__main__":
    # create_collection()  # Uncomment if you want to create the collection
    csv_list_file = [
        # '1-0-เรียนล่วงหน้า.csv',
        # '1-1-ช้างเผือก.csv',
        # '1-1-นานาชาติและภาษาอังกฤษ.csv',
        '1-1-รับนักกีฬาดีเด่น.csv',
        # '1-2-ช้างเผือก.csv',
        # '1-2-โอลิมปิกวิชาการ.csv',
        # '2-0-MOU.csv',
        # '2-0-โควต้า30จังหวัด.csv',
        # '2-0-เพชรนนทรี.csv',
        # '2-0-ลูกพระพิรุณ.csv',
        # '2-0-นานาชาติและภาษาอังกฤษ.csv',
        # '2-0-ผู้มีความสามารถทางกีฬา.csv',
        # '2-0-นักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์.csv',
        # '3-0-Admission.csv',
    ]
    for file in csv_list_file:
        df = read_csv_data(f"output/{file}") 
        process_and_insert_data(df)
