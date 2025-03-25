import chromadb
from openai import OpenAI

def embed_document(document, collection_name):
    # Connect to ChromaDB
    client = chromadb.Client()
    # Create collection
    collection = client.create_collection(collection_name)
    # Embed document
    openai_client = OpenAI()
    response = openai_client.embeddings.create(input=document, model='text-embedding-ada-002')
    embedding = response.data[0].embedding
    # Store in vector database
    collection.add(documents=[document], embeddings=[embedding], ids=['doc1'])
    return True