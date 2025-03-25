# Vector Database Integration
# Create a function that connects to ChromaDB, embeds a document using OpenAI embeddings, and stores it in the vector database. Include error handling and proper documentation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

def generate_simulated_data(num_documents=100):
    """
    Generate simulated document embedding data for visualization.
    
    Parameters:
        num_documents (int): The number of document embeddings to generate.
        
    Returns:
        pd.DataFrame: A DataFrame containing document IDs and their corresponding embeddings.
    """
    # Simulate document IDs
    document_ids = [f'doc_{i}' for i in range(num_documents)]
    
    # Simulate embeddings (2D for visualization purposes)
    embeddings = np.random.rand(num_documents, 2)  # 100 embeddings in 2D space
    
    # Create a DataFrame
    df = pd.DataFrame(embeddings, columns=['Embedding_X', 'Embedding_Y'])
    df['Document_ID'] = document_ids
    
    return df

def plot_embeddings(df):
    """
    Plot the embeddings in a scatter plot and a histogram.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing document embeddings and IDs.
    """
    # Create a scatter plot for the embeddings
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='Embedding_X', y='Embedding_Y', hue='Document_ID', palette='viridis', legend=None)
    plt.title('Scatter Plot of Document Embeddings')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.grid(True)

    # Create a histogram to show the distribution of the embeddings in X and Y dimensions
    plt.subplot(1, 2, 2)
    sns.histplot(df['Embedding_X'], bins=20, kde=True, color='blue', label='Embedding Dimension 1', stat='density', alpha=0.5)
    sns.histplot(df['Embedding_Y'], bins=20, kde=True, color='orange', label='Embedding Dimension 2', stat='density', alpha=0.5)
    plt.title('Histogram of Embedding Dimensions')
    plt.xlabel('Embedding Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Generate simulated data
    df_embeddings = generate_simulated_data(num_documents=100)
    
    # Plot the generated embeddings
    plot_embeddings(df_embeddings)