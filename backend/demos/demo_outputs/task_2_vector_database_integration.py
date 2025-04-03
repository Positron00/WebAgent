# Vector Database Integration
# Create a function that connects to ChromaDB, embeds a document using OpenAI embeddings, and stores it in the vector database. Include error handling and proper documentation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chromadb import Client  # Assuming ChromaDB has a client library
from openai import OpenAI  # Assuming there's an OpenAI library to handle embeddings

# Set the aesthetics for seaborn
sns.set(style='whitegrid')

def embed_and_store_document(document, db_url, openai_api_key):
    """
    Connects to ChromaDB and embeds a document using OpenAI embeddings,
    then stores it in the vector database.
    
    Parameters:
    - document (str): The document to be embedded and stored.
    - db_url (str): The URL of the ChromaDB instance.
    - openai_api_key (str): The API key for OpenAI embeddings.

    Returns:
    - bool: True if the document was stored successfully, False otherwise.
    """
    try:
        # Connect to ChromaDB
        client = Client(db_url)
        
        # Initialize OpenAI with the provided API key
        openai.api_key = openai_api_key
        
        # Get the embedding for the document
        embedding = OpenAI.Embedding.create(input=document)
        vector = embedding['data'][0]['embedding']
        
        # Store the vector in the database
        client.store_vector(vector)
        
        return True
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

def visualize_storage_results(success_count, failure_count):
    """
    Visualizes the results of document storage in ChromaDB using a bar chart.
    
    Parameters:
    - success_count (int): The number of documents successfully stored.
    - failure_count (int): The number of documents that failed to store.
    """
    # Data for visualization
    categories = ['Success', 'Failure']
    counts = [success_count, failure_count]
    
    # Create a DataFrame for better plotting
    df = pd.DataFrame({'Category': categories, 'Count': counts})
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Category', y='Count', data=df, palette='viridis')
    
    # Adding titles and labels
    plt.title('Document Storage Results', fontsize=16)
    plt.xlabel('Storage Status', fontsize=14)
    plt.ylabel('Number of Documents', fontsize=14)
    
    # Display the plot
    plt.show()

# Example usage of the function
if __name__ == "__main__":
    document_text = "This is an example document text to embed."
    db_url = "YOUR_CHROMADB_URL"  # Replace with your actual ChromaDB URL
    openai_api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key

    # Attempt to embed and store the document
    success = embed_and_store_document(document_text, db_url, openai_api_key)
    
    # Visualize the results
    if success:
        visualize_storage_results(success_count=1, failure_count=0)
    else:
        visualize_storage_results(success_count=0, failure_count=1)