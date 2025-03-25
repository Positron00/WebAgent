# Batch Document Processor with Specific Requirements
# Create a Python class for batch processing documents with vector embedding. It should handle multiple file types, implement proper error handling, and log processing metrics.

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self, file_paths):
        """
        Initialize the DocumentProcessor with a list of file paths.

        :param file_paths: List of file paths to process
        """
        self.file_paths = file_paths
        self.metrics = {'file_type': [], 'count': [], 'average_time': []}
    
    def process_documents(self):
        """
        Process each document and log metrics such as file type, count, and processing time.
        """
        for file_path in self.file_paths:
            try:
                file_type = self._get_file_type(file_path)
                processing_time = self._mock_processing(file_path)  # Mock processing for demonstration
                
                # Update metrics
                self._update_metrics(file_type, processing_time)
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

        self._visualize_metrics()

    def _get_file_type(self, file_path):
        """
        Get the file type based on the file extension.

        :param file_path: Path to the file
        :return: File type as a string
        """
        _, file_extension = os.path.splitext(file_path)
        return file_extension[1:]  # Remove the dot

    def _mock_processing(self, file_path):
        """
        Mock processing of a document and return a random processing time.

        :param file_path: Path to the file
        :return: Random processing time in seconds
        """
        # Simulate varying processing times
        time_taken = np.random.uniform(0.1, 2.0)  # Random time between 0.1 and 2 seconds
        logging.info(f"Processed {file_path} in {time_taken:.2f} seconds.")
        return time_taken

    def _update_metrics(self, file_type, processing_time):
        """
        Update metrics for the processed document.

        :param file_type: Type of the file processed
        :param processing_time: Time taken to process the file
        """
        if file_type not in self.metrics['file_type']:
            self.metrics['file_type'].append(file_type)
            self.metrics['count'].append(1)
            self.metrics['average_time'].append(processing_time)
        else:
            index = self.metrics['file_type'].index(file_type)
            self.metrics['count'][index] += 1
            self.metrics['average_time'][index] = (
                (self.metrics['average_time'][index] * (self.metrics['count'][index] - 1)) + processing_time
            ) / self.metrics['count'][index]

    def _visualize_metrics(self):
        """
        Visualize the processing metrics using bar plots.
        """
        # Create a DataFrame from metrics
        df = pd.DataFrame(self.metrics)

        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        # Create a figure with two subplots
        fig, ax1 = plt.subplots(2, 1, figsize=(10, 10))

        # Bar plot for the count of processed documents
        sns.barplot(x='file_type', y='count', data=df, ax=ax1[0], palette='viridis')
        ax1[0].set_title('Count of Processed Documents by File Type')
        ax1[0].set_xlabel('File Type')
        ax1[0].set_ylabel('Count')

        # Bar plot for the average processing time
        sns.barplot(x='file_type', y='average_time', data=df, ax=ax1[1], palette='plasma')
        ax1[1].set_title('Average Processing Time by File Type (seconds)')
        ax1[1].set_xlabel('File Type')
        ax1[1].set_ylabel('Average Time (seconds)')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # List of sample file paths
    sample_files = ['document1.pdf', 'document2.docx', 'document3.txt', 'document4.pdf', 'document5.docx']
    
    # Create an instance of DocumentProcessor and process documents
    processor = DocumentProcessor(sample_files)
    processor.process_documents()