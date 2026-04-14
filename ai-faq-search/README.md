# AI FAQ Semantic Search System

## 📌 Overview

This project is a simple AI-based semantic search system that retrieves relevant answers for user questions using vector embeddings.

It understands the meaning of the question and returns the most appropriate answer instead of matching exact keywords.

---

## 🎯 Objective

* Build a semantic search system
* Retrieve answers based on meaning
* Demonstrate vector search concepts

---

## ⚙️ Tech Stack

* Python
* Sentence Transformers
* NumPy

---

## 📂 Dataset

The dataset contains a small set of frequently asked questions and answers such as:

* What is AI
* What is machine learning
* What is Python used for
* What is data science

---

## 🧠 How it Works

1. Convert FAQ questions into vector embeddings
2. Store embeddings (simulating Endee vector database)
3. Convert user query into embedding
4. Compare similarity using cosine similarity
5. Return the best matching answer

---

## 🔥 Features

* Semantic question answering
* Fast retrieval
* Simple chatbot interface
* Similarity-based matching

---

## 🚀 How to Run

pip install sentence-transformers numpy
python rag_app.py

---

## 📊 Sample Output

**Input:** What is AI?
**Output:** Artificial Intelligence is the simulation of human intelligence.

**Input:** What is machine learning?
**Output:** Machine learning is a subset of AI that learns from data.

**Input:** What is Python used for?
**Output:** Python is used for web development, AI, and data science.

---

## ⚠️ Endee Integration

This project follows Endee vector database concepts:

* Vector storage
* Similarity search
* Fast retrieval

(Note: Implemented as a prototype using in-memory storage)

---

## 🏁 Conclusion

This project demonstrates how vector embeddings can be used to build a simple semantic search system.
