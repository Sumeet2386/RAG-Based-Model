Audio → Searchable Knowledge (Local RAG)
This project turns long audio lectures into a searchable system where you can ask questions and get answers based only on the lecture content.

Everything runs locally using Ollama. No external APIs.

What it does :-
Converts audio to text using Whisper
Cleans and splits the transcript into chunks
Creates embeddings using bge-m3
Stores vectors locally for similarity search
Retrieves relevant chunks for a question
Uses a Llama model to generate the final answer
Simple Streamlit UI for interaction

Stack :-
Python
Whisper
Ollama (bge-m3, llama3)
FAISS (vector store)
Streamlit

How it works

Audio file
→ Transcription
→ Text chunking
→ Embeddings
→ Vector storage
→ Question
→ Retrieval
→ LLM answer

Run locally

Install dependencies:-
pip install -r requirements.txt


Pull models:
ollama pull bge-m3
ollama pull llama3


Ingest data:-
python ingest.py


Start the app:-
streamlit run app.py

