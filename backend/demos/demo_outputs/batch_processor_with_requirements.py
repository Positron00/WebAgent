# Batch Document Processor with Specific Requirements
# Create a Python class for batch processing documents with vector embedding. It should handle multiple file types, implement proper error handling, and log processing metrics.

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self, file_paths: List[str]):
        """
        Initializes the DocumentProcessor with a list of file paths.
        
        Args:
            file_paths (List[str]): List of paths to the documents to process.
        """
        self.file_paths = file_paths
        self.processed_data = []
        self.metrics = {'file_type': [], 'processing_time': []}

    def process_documents(self):
        """
        Processes the documents at the specified file paths, logging metrics for each file type.
        """
        for file_path in self.file_paths:
            file_type = self.get_file_type(file_path)
            start_time = time()
            try:
                self.process_file(file_path)
                processing_time = time() - start_time
                self.record_metrics(file_type, processing_time)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

    def get_file_type(self, file_path: str) -> str:
        """
        Retrieves the file type from the file extension.
        
        Args:
            file_path (str): The path to the file.
        
        Returns:
            str: The file type based on the file extension.
        """
        return os.path.splitext(file_path)[1][1:]

    def process_file(self, file_path: str):
        """
        A placeholder method for processing the file. Replace with actual processing logic.
        
        Args:
            file_path (str): The path to the file to process.
        """
        # Simulating processing time
        if file_path.endswith('.txt'):
            time.sleep(np.random.uniform(0.1, 0.5))
        elif file_path.endswith('.csv'):
            time.sleep(np.random.uniform(0.2, 0.7))
        elif file_path.endswith('.pdf'):
            time.sleep(np.random.uniform(0.3, 0.9))
        else:
            raise ValueError("Unsupported file type")

    def record_metrics(self, file_type: str, processing_time: float):
        """
        Records the metrics for processed files.
        
        Args:
            file_type (str): The type of the file processed.
            processing_time (float): The time taken to process the file.
        """
        self.metrics['file_type'].append(file_type)
        self.metrics['processing_time'].append(processing_time)
        logging.info(f"Processed {file_type} in {processing_time:.2f} seconds")

    def visualize_metrics(self):
        """
        Visualizes the processing metrics using bar charts and boxplots.
        """
        df = pd.DataFrame(self.metrics)

        # Bar chart for number of documents processed per file type
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='file_type', order=df['file_type'].value_counts().index)
        plt.title('Number of Documents Processed per File Type')
        plt.xlabel('File Type')
        plt.ylabel('Count of Documents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Boxplot for processing time per file type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='file_type', y='processing_time')
        plt.title('Distribution of Processing Time per File Type')
        plt.xlabel('File Type')
        plt.ylabel('Processing Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Sample file paths for demonstration
    sample_files = [
        'document1.txt',
        'document2.csv',
        'document3.pdf',
        'document4.txt',
        'document5.csv'
    ]
    
    processor = DocumentProcessor(sample_files)
    processor.process_documents()
    processor.visualize_metrics()