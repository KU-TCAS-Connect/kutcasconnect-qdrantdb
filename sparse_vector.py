from FlagEmbedding import BGEM3FlagModel

# Initialize the model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Your Thai text sentence
sentences_1 = ["วิศวกรรมไฟฟ้า ภาคพิเศษ รอบ3 มีเกณฑ์การรับเป็นอย่างไรบ้างและรับกี่คนคะ"]

# Encode the sentences using the BGEM3 model (returning dense and sparse vectors)
output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# Print the structure of the output to explore available keys
print(output_1)

# If sparse vector is under a different key, you can extract it accordingly.
# For example, if it's under 'lexical_weights', you can access it like this:
if 'lexical_weights' in output_1:
    sparse_vector = output_1['lexical_weights'][0]
    print("Lexical Weights:", sparse_vector)
else:
    print("No sparse vector found under 'lexical_weights'. Please inspect the keys.")

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

sentences_1 = ["รอบการคัดเลือก: 1 โครงการ: เรียนล่วงหน้า สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมการบินและอวกาศ จำนวนรับ: 5 เงื่อนไขขั้นต่ำ: 1. เป็นนักเรียนชั้น ม.6 สายสามัญ หรือสายอาชีวะ ที่เข้าร่วมโครงการเรียนล่วงหน้าของมหาวิทยาลัยเกษตรศาสตร์ เกณฑ์การพิจารณา: 1. เลือก 2 วิชา จาก คณิตศาสตร์, ฟิสิกส์, คอมพิวเตอร์และการโปรแกรม, การเขียนแบบวิศวกรรม โดยที่เกรดแต่ละวิชา ≥ B"]

output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# you can see the weight for each token:
sparse = model.convert_id_to_token(output_1['lexical_weights'])
print(model.convert_id_to_token(output_1['lexical_weights']))
# [{'What': 0.08356, 'is': 0.0814, 'B': 0.1296, 'GE': 0.252, 'M': 0.1702, '3': 0.2695, '?': 0.04092},
#  {'De': 0.05005, 'fin': 0.1368, 'ation': 0.04498, 'of': 0.0633, 'BM': 0.2515, '25': 0.3335}]

from FlagEmbedding import BGEM3FlagModel

# Initialize the BGEM3 model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # Use FP16 for faster computation

# Your Thai text sentence
sentences_1 = ["รอบการคัดเลือก: 1 โครงการ: เรียนล่วงหน้า สาขาวิชา: วศ.บ. สาขาวิชาวิศวกรรมการบินและอวกาศ จำนวนรับ: 5 เงื่อนไขขั้นต่ำ: 1. เป็นนักเรียนชั้น ม.6 สายสามัญ หรือสายอาชีวะ ที่เข้าร่วมโครงการเรียนล่วงหน้าของมหาวิทยาลัยเกษตรศาสตร์ เกณฑ์การพิจารณา: 1. เลือก 2 วิชา จาก คณิตศาสตร์, ฟิสิกส์, คอมพิวเตอร์และการโปรแกรม, การเขียนแบบวิศวกรรม โดยที่เกรดแต่ละวิชา ≥ B"]

# Encode the sentence using the BGEM3 model (returning dense and sparse vectors)
output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# Extract the lexical weights (this is your sparse representation)
lexical_weights = output_1['lexical_weights'][0]

# Convert the lexical weights into a dictionary (index: weight)
sparse_vector_dict = {token: weight for token, weight in lexical_weights.items()}

# Print the sparse vector dictionary
print("Sparse vector dictionary for Qdrant:", sparse_vector_dict)

indices = list(sparse_vector_dict.keys())  # Indices of the sparse vector
values = list(sparse_vector_dict.values())  # Values of the sparse vector

print(indices)

print(values)
