from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Using Endee vector database (simulated)...")

# FAQ Dataset
questions = [
    "What is AI?",
    "What is machine learning?",
    "What is deep learning?",
    "What is Python used for?",
    "What is data science?",
    "What is cloud computing?"
]

answers = [
    "Artificial Intelligence is the simulation of human intelligence.",
    "Machine learning is a subset of AI that learns from data.",
    "Deep learning uses neural networks for complex tasks.",
    "Python is used for web development, AI, and data science.",
    "Data science involves analyzing data to gain insights.",
    "Cloud computing provides computing services over the internet."
]

# Convert questions into vectors
question_embeddings = model.encode(questions)

def search(query):
    query_embedding = model.encode([query])
    scores = np.dot(question_embeddings, query_embedding.T).flatten()
    best_match = np.argmax(scores)
    return answers[best_match], scores[best_match]

# Chat loop
while True:
    query = input("\nAsk a question (type 'exit'): ")
    
    if query.lower() == "exit":
        break
    
    result, score = search(query)
    
    print("\nAnswer:", result)
    print("Score:", score)
