# Batch Document Processor with Specific Requirements
# Create a Python class for batch processing documents with vector embedding. It should handle multiple file types, implement proper error handling, and log processing metrics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetics for seaborn
sns.set(style="whitegrid")

class DocumentBatchProcessor:
    def __init__(self, data):
        """
        Initialize the DocumentBatchProcessor with data.
        
        :param data: A DataFrame containing file types, processing times, and success flags.
        """
        self.data = data
        self.metrics = None

    def process_documents(self):
        """
        Process documents and calculate metrics.
        """
        try:
            # Calculate the number of files processed by type
            self.metrics = self.data.groupby('file_type').agg(
                total_files=('file_type', 'count'),
                success_rate=('success', 'mean'),
                average_processing_time=('processing_time', 'mean')
            ).reset_index()
            print("Processing completed successfully.")
        except Exception as e:
            print(f"Error during processing: {e}")

    def visualize_metrics(self):
        """
        Create visualizations for the metrics.
        """
        if self.metrics is None:
            print("No metrics available. Please run process_documents first.")
            return

        # Create a bar plot for total files processed by type
        plt.figure(figsize=(10, 5))
        sns.barplot(data=self.metrics, x='file_type', y='total_files', palette='viridis')
        plt.title('Total Files Processed by Type')
        plt.xlabel('File Type')
        plt.ylabel('Total Files')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Create a line plot for average processing time by type
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=self.metrics, x='file_type', y='average_processing_time', marker='o')
        plt.title('Average Processing Time by File Type')
        plt.xlabel('File Type')
        plt.ylabel('Average Processing Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Sample Data
data = {
    'file_type': ['pdf', 'docx', 'pdf', 'txt', 'docx', 'pdf'],
    'processing_time': [1.2, 0.8, 1.0, 0.5, 1.4, 1.3],
    'success': [1, 1, 1, 0, 1, 1]  # 1 indicates success, 0 indicates failure
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create an instance of DocumentBatchProcessor and process the data
processor = DocumentBatchProcessor(df)
processor.process_documents()
processor.visualize_metrics()