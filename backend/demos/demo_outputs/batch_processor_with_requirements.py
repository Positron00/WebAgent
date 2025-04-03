# Batch Document Processor with Specific Requirements
# Create a Python class for batch processing documents with vector embedding. It should handle multiple file types, implement proper error handling, and log processing metrics.

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentBatchProcessor:
    def __init__(self, file_paths):
        """
        Initialize the DocumentBatchProcessor with file paths.
        
        Parameters:
        file_paths (list): List of file paths to process.
        """
        self.file_paths = file_paths
        self.metrics = {
            'file_type': [],
            'process_time': [],
            'status': []
        }

    def process_files(self):
        """Process each file and log metrics."""
        for file_path in self.file_paths:
            try:
                # Check if file exists
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Simulate processing time
                start_time = datetime.now()
                file_type = self.get_file_type(file_path)
                self.simulate_embedding_process(file_path)  # Placeholder for actual embedding logic
                end_time = datetime.now()
                
                process_time = (end_time - start_time).total_seconds()
                self.log_metrics(file_type, process_time, 'Processed successfully')

            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                self.log_metrics(file_type, 0, 'Error')

    def get_file_type(self, file_path):
        """Get the file type based on the file extension."""
        return os.path.splitext(file_path)[1][1:]  # Return file extension without dot

    def simulate_embedding_process(self, file_path):
        """Simulate embedding processing (replace with actual logic)."""
        # Simulating a delay for processing
        import time
        time.sleep(np.random.uniform(0.1, 0.5))  # Simulating variable processing time

    def log_metrics(self, file_type, process_time, status):
        """Log processing metrics."""
        self.metrics['file_type'].append(file_type)
        self.metrics['process_time'].append(process_time)
        self.metrics['status'].append(status)

    def visualize_metrics(self):
        """Visualize the processing metrics."""
        df = pd.DataFrame(self.metrics)

        # Filter successful processes for visualization
        success_df = df[df['status'] == 'Processed successfully']

        # Create bar plot for number of documents processed by file type
        plt.figure(figsize=(10, 6))
        sns.countplot(data=success_df, x='file_type', palette='viridis')
        plt.title("Number of Documents Processed by File Type")
        plt.xlabel("File Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Create bar plot for average processing time by file type
        avg_time = success_df.groupby('file_type')['process_time'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_time, x='file_type', y='process_time', palette='coolwarm')
        plt.title("Average Processing Time by File Type")
        plt.xlabel("File Type")
        plt.ylabel("Average Processing Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Sample usage
if __name__ == "__main__":
    # Sample data: Replace with actual file paths
    sample_files = [
        "document_1.txt",
        "document_2.pdf",
        "document_3.docx",
        "non_existent_file.txt"
    ]

    # Create an instance of the DocumentBatchProcessor
    processor = DocumentBatchProcessor(sample_files)
    
    # Process files
    processor.process_files()
    
    # Visualize the metrics
    processor.visualize_metrics()