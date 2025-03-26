# Vector Database Integration
# Create a function that connects to ChromaDB, embeds a document using OpenAI embeddings, and stores it in the vector database. Include error handling and proper documentation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to connect to ChromaDB, embed documents, and store them in the database
def embed_and_store_documents(documents):
    """
    Connects to ChromaDB, embeds the provided documents using OpenAI embeddings,
    and stores them in the vector database.

    Parameters:
    documents (list): A list of documents to embed and store.

    Returns:
    pd.DataFrame: A DataFrame containing the documents and their embeddings.
    """
    import chromadb
    from openai import OpenAI

    # Create mock embeddings for demonstration
    embeddings = []
    for doc in documents:
        try:
            # Simulate an embedding function (this should call OpenAI's actual embedding API)
            embedding = np.random.rand(5)  # Replace this with actual embedding API call
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error embedding document '{doc}': {e}")
            embeddings.append(np.nan * 5)

    # Create a DataFrame to hold documents and their embeddings
    df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(1, 6)])
    df['document'] = documents

    # Store embeddings in ChromaDB (mock implementation)
    try:
        # Assuming a ChromaDB client is created here
        client = chromadb.Client()
        collection = client.get_collection("documents")

        for index, row in df.iterrows():
            collection.add(documents=[row['document']], embeddings=[row[['embedding_1', 'embedding_2', 'embedding_3', 'embedding_4', 'embedding_5']].values.tolist()])
    except Exception as e:
        print(f"Error storing documents in ChromaDB: {e}")

    return df

# Function to visualize the embeddings
def visualize_embeddings(df):
    """
    Visualizes the embeddings of the documents using scatter plot and histogram.

    Parameters:
    df (pd.DataFrame): DataFrame containing document embeddings.
    """
    plt.figure(figsize=(12, 6))

    # Scatter Plot
    plt.subplot(1, 2, 1)
    plt.scatter(df['embedding_1'], df['embedding_2'], alpha=0.7, c='blue', edgecolors='k')
    plt.title('Scatter Plot of Document Embeddings')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.grid(True)

    # Histogram of Embeddings
    plt.subplot(1, 2, 2)
    sns.histplot(df[['embedding_1', 'embedding_2']].values.flatten(), bins=30, kde=True, color='purple')
    plt.title('Histogram of Embedding Values')
    plt.xlabel('Embedding Values')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Mock documents for embedding
    mock_documents = [
        "Document 1: The quick brown fox jumps over the lazy dog.",
        "Document 2: A journey of a thousand miles begins with a single step.",
        "Document 3: To be or not to be, that is the question.",
        "Document 4: All that glitters is not gold.",
        "Document 5: The only thing we have to fear is fear itself."
    ]
    
    # Embed and store documents
    embeddings_df = embed_and_store_documents(mock_documents)
    
    # Visualize the embeddings
    visualize_embeddings(embeddings_df)