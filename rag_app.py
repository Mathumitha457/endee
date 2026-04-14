from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Artificial Intelligence is the future.",
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Python is used for AI development.",
    "AI is used in healthcare and finance.",
    "Data science helps in decision making."
]

doc_embeddings = model.encode(documents)

def search(query):
    query_embedding = model.encode([query])
    scores = np.dot(doc_embeddings, query_embedding.T)
    best_match = np.argmax(scores)
    return documents[best_match]

while True:
    query = input("Ask something: ")
    if query == "exit":
        break
    result = search(query)
    print("Answer:", result)
