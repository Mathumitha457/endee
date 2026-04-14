from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Using Endee vector database (simulated)...")

# Dataset (job profiles)
documents = [
    "Java developer with experience in Spring Boot",
    "Machine learning engineer with Python and TensorFlow",
    "Frontend developer with React and JavaScript",
    "Backend developer with Node.js and MongoDB",
    "Data analyst with SQL and Power BI",
    "AI engineer working on NLP and deep learning",
    "Software engineer with C++ and system design",
    "Cloud engineer with AWS and Docker"
]

# Convert documents to vectors
doc_embeddings = model.encode(documents)

# Search function
def search(query):
    query_embedding = model.encode([query])
    
    # Cosine similarity
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    
    best_match_index = np.argmax(scores)
    
    return documents[best_match_index], scores[best_match_index]

# Chat loop
while True:
    query = input("\nAsk something (type 'exit' to stop): ")
    
    if query.lower() == "exit":
        print("Exiting...")
        break
    
    result, score = search(query)
    
    print("\n🔍 Best Match:")
    print(result)
    print(f"🔢 Similarity Score: {score:.2f}")
