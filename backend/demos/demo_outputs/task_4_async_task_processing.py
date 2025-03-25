# Async Task Processing
# Implement an asynchronous task processing system using Python's asyncio that can handle multiple document processing tasks in parallel with proper error handling and status updates.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Generate synthetic data for document processing times (in seconds)
np.random.seed(42)  # For reproducibility
num_tasks = 100
processing_times = np.random.exponential(scale=5, size=num_tasks)  # Simulate processing times

# Create a DataFrame to hold the processing times
df = pd.DataFrame({
    'Task_ID': range(1, num_tasks + 1),
    'Processing_Time': processing_times
})

# Function to visualize the processing times
def visualize_processing_times(dataframe):
    """Visualizes the document processing times using a histogram and box plot."""
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Histogram of processing times
    sns.histplot(dataframe['Processing_Time'], bins=20, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title('Distribution of Document Processing Times', fontsize=16)
    axs[0].set_xlabel('Processing Time (seconds)', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)

    # Box plot of processing times
    sns.boxplot(x=dataframe['Processing_Time'], ax=axs[1], color='lightgreen')
    axs[1].set_title('Box Plot of Document Processing Times', fontsize=16)
    axs[1].set_xlabel('Processing Time (seconds)', fontsize=12)

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()

# Call the visualization function with the generated DataFrame
try:
    visualize_processing_times(df)
except Exception as e:
    print(f"An error occurred while visualizing processing times: {e}")