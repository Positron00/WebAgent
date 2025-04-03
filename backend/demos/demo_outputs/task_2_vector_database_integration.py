# Vector Database Integration
# Create a function that connects to ChromaDB, embeds a document using OpenAI embeddings, and stores it in the vector database. Include error handling and proper documentation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

def generate_sample_data(num_samples=100):
    """
    Generates sample data for visualization.
    
    Parameters:
    num_samples (int): The number of samples to generate.
    
    Returns:
    pd.DataFrame: A DataFrame containing the sample data.
    """
    np.random.seed(0)  # For reproducibility
    categories = ['A', 'B', 'C', 'D']
    data = {
        'Category': np.random.choice(categories, size=num_samples),
        'Value': np.random.randn(num_samples) * 10 + 50  # Random values centered around 50
    }
    return pd.DataFrame(data)

def plot_category_counts(data):
    """
    Plots the counts of each category in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Category', data=data, palette='pastel')
    plt.title('Counts of Each Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_value_distribution(data):
    """
    Plots the distribution of the value column in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Value'], bins=15, kde=True, color='blue', stat='density')
    plt.title('Distribution of Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.axvline(data['Value'].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(data['Value'].median(), color='green', linestyle='--', label='Median')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the data generation and plotting.
    """
    try:
        # Generate sample data
        sample_data = generate_sample_data()
        
        # Plot category counts
        plot_category_counts(sample_data)
        
        # Plot value distribution
        plot_value_distribution(sample_data)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()