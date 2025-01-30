import uuid
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams
import ast
import openai
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from typing import List

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

client = QdrantClient("http://localhost:6333")
collection_name = "ku_tcas_document_bge"

admission_round = "3"
admission_program = "Admission"

# Initialize the BGEM3 model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # Use FP16 for faster computation

st_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create a collection (if not already created)
def create_collection():
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # Size for the dense vector (for example)
            distance=Distance.COSINE  # Choose from COSINE, EUCLIDEAN, DOT
        ),
        sparse_vectors_config={
            "keywords": models.SparseVectorParams(  # Field name for sparse vectors
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"Collection '{collection_name}' created successfully.")

def uuid_from_time(timestamp, index):
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{timestamp.isoformat()}_{index}")

# def chunk_text(text, chunk_size=512, overlap=64):
#     words = text.split()
#     chunks = []
#     start = 0

#     while start < len(words):
#         chunk = " ".join(words[start:start + chunk_size])
#         chunks.append(chunk)
#         start += chunk_size - overlap  # Overlapping to maintain context

#     return chunks

def chunk_text(text, num_chunks=5):
    words = text.split()
    total_words = len(words)
    chunk_size = total_words // num_chunks  # Divide the total words evenly among the chunks
    chunks = []

    start = 0
    for i in range(num_chunks):
        end = start + chunk_size
        if i == num_chunks - 1:  # Ensure the last chunk includes all remaining words
            end = total_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end

    return chunks

def process_and_insert_data(df, admission_info):
    points = []
    for _, row in df.iterrows():
        content = row['content']  # Assuming 'content' is the column with text from your .txt files
        filename = row['filename']  # Add the filename to the row (assuming 'filename' is part of the DataFrame)

        # Extract the admission info from the passed dictionary
        admission_round = admission_info['admission_round']
        admission_program = admission_info['admission_program']
        reference = admission_info['reference']

        # Chunk text
        chunks = chunk_text(content, num_chunks=5)

        # Use BGEM3 to generate dense and sparse vectors
        output_1 = model.encode(chunks, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        for i, chunk in enumerate(chunks):
            lexical_weights = output_1['lexical_weights'][i]
            sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

            # Use BGEM3 dense vector for embedding
            dense_vector = output_1['dense_vecs'][i]

            data = {
                "id": str(uuid_from_time(datetime.now(), i)),
                "metadata": {
                    "admission_round": admission_round,
                    "admission_program": admission_program,
                    "reference": reference,
                    "created_at": datetime.now().isoformat(),
                    "filename": filename,  # Add the filename to the metadata
                    "chunk_number": i + 1,
                },
                "contents": chunk,
                "embedding": dense_vector,  # Use BGEM3 dense vector
            }

            indices = list(sparse_vector_dict.keys())
            values = [float(x) for x in sparse_vector_dict.values()]
            sparse_vector = dict(zip(indices, values))

            point = PointStruct(
                id=data["id"],
                vector={ 
                    "": data["embedding"],  # Dense vector from BGEM3
                    "keywords": models.SparseVector(
                        indices=indices,
                        values=values
                    ),
                },
                payload={ 
                    "admission_round": data["metadata"]["admission_round"],
                    "admission_program": data["metadata"]["admission_program"],
                    "reference": data["metadata"]["reference"],
                    "created_at": data["metadata"]["created_at"],
                    "filename": data["metadata"]["filename"],  # Add filename to the payload
                    "chunk_number": data["metadata"]["chunk_number"],  # Add chunk number to the payload
                    "contents": data["contents"],
                    "lexical_weights": sparse_vector,
                },
            )
            
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

    num_chunks = 5 # how many chunk we want?

    admission_info_mapping = {
        '68-TCAS1-AP.txt': {
            'admission_round': "1",
            'admission_program': "โครงการเรียนล่วงหน้า",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/11/11/68_TCAS1_AP_1.1_edit.pdf"
        },
        '68-TCAS1-International_Program.txt': {
            'admission_round': "1",
            'admission_program': "นานาชาติและภาษาอังกฤษ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/11/68-TCAS1-International_Program_1.1.pdf"
        },
        '68-TCAS1-Olympics.txt': {
            'admission_round': "1",
            'admission_program': "โอลิมปิกวิชาการ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/16/68_TCAS1_Oiympics_1.1.pdf"
        },
        '68-TCAS1-Sport.txt': {
            'admission_round': "1",
            'admission_program': "โครงการรับนักกีฬาดีเด่น",
            'reference':"https://admission.ku.ac.th/media/announcements/2024/10/03/68-TCAS1-Sport_1.1.pdf"
        },
        '68-TCAS1-White_Elephant.txt': {
            'admission_round': "1",
            'admission_program': "โครงการช้างเผือก",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/02/68_TCAS1_White_Elephant_1.1.pdf"
        },
        '68-TCAS2-Diamond_Nontri.txt': {
            'admission_round': "1",
            'admission_program': "โครงการเพชรนนทรี",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/12/26/68-TCAS2-Diamond_Nontri.pdf"
        },
        '68-TCAS2-International_Program.txt': {
            'admission_round': "2",
            'admission_program': "นานาชาติและภาษาอังกฤษ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/18/68-TCAS2-International_Program.pdf"
        },
        '68-TCAS2-KU_MOU.txt': {
            'admission_round': "2",
            'admission_program': "โครงการความร่วมมือในการสร้างเครือข่ายทางการศึกษากับมหาวิทยาลัยเกษตรศาสตร์",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/25/68-TCAS2-KU_MOU.pdf"
        },
        '68-TCAS2-Pra_Pirun.txt': {
            'admission_round': "2",
            'admission_program': "โครงการลูกพระพิรุณ",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Pra_Pirun.pdf"
        },
        '68-TCAS2-Province_30.txt': {
            'admission_round': "2",
            'admission_program': "โควตา30จังหวัด",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Province_30.pdf"
        },
        '68-TCAS2-Satit.txt': {
            'admission_round': "2",
            'admission_program': "นักเรียนดีเด่นจากโรงเรียนสาธิตแห่งมหาวิทยาลัยเกษตรศาสตร์",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Province_30.pdf"
        },
        '68-TCAS2-Sport.txt': {
            'admission_round': "2",
            'admission_program': "ผู้มีความสามารถทางกีฬา",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/11/11/68-TCAS2-Sport.pdf"
        },
        '68-TCAS3-Admission.txt': {
            'admission_round': "3",
            'admission_program': "Admission",
            'reference': "https://admission.ku.ac.th/media/announcements/2024/10/31/68-TCAS3-Admission_edit-1.pdf"
        }
    }
    
    txt_list_file = [
        '68-TCAS1-AP.txt',
        '68-TCAS1-International_Program.txt',
        '68-TCAS1-Olympics.txt',
        '68-TCAS1-Sport.txt',
        '68-TCAS1-White_Elephant.txt',
        '68-TCAS2-Diamond_Nontri.txt',
        '68-TCAS2-International_Program.txt',
        '68-TCAS2-KU_MOU.txt',
        '68-TCAS2-Pra_Pirun.txt',
        '68-TCAS2-Province_30.txt',
        '68-TCAS2-Satit.txt',
        '68-TCAS2-Sport.txt',
        '68-TCAS3-Admission.txt',
    ]

    # reference_list = [
        # "https://admission.ku.ac.th/media/announcements/2024/11/11/68_TCAS1_AP_1.1_edit.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/11/68-TCAS1-International_Program_1.1.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/16/68_TCAS1_Oiympics_1.1.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/03/68-TCAS1-Sport_1.1.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/02/68_TCAS1_White_Elephant_1.1.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/12/26/68-TCAS2-Diamond_Nontri.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/18/68-TCAS2-International_Program.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/25/68-TCAS2-KU_MOU.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Pra_Pirun.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Province_30.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/15/68-TCAS2-Satit.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/11/11/68-TCAS2-Sport.pdf",
        # "https://admission.ku.ac.th/media/announcements/2024/10/31/68-TCAS3-Admission_edit-1.pdf",
    # ]

    for file in txt_list_file:
        admission_info = admission_info_mapping.get(file)
        if admission_info:  # Check if the file is mapped to admission info
            with open(f"output/txt/content/{file}", "r", encoding="utf-8") as f:
                content = f.read()
            
            # Assuming you want to process the content as part of the DataFrame
            df = pd.DataFrame({'content': [content], 'filename': [file]})
            process_and_insert_data(df, admission_info)
        else:
            print(f"No admission info found for {file}")
