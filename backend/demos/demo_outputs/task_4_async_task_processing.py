# Async Task Processing
# Implement an asynchronous task processing system using Python's asyncio that can handle multiple document processing tasks in parallel with proper error handling and status updates.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Generate sample data for document processing tasks
np.random.seed(42)  # For reproducibility
num_tasks = 100
statuses = np.random.choice(['Success', 'Failed'], size=num_tasks, p=[0.8, 0.2])
processing_times = np.random.exponential(scale=2, size=num_tasks)  # Simulating processing times in seconds

# Create a DataFrame
data = pd.DataFrame({
    'Status': statuses,
    'Processing Time (s)': processing_times
})

# Display the first few rows of the DataFrame
print("Sample Data:")
print(data.head())

# Visualization 1: Bar plot of task statuses
plt.figure(figsize=(10, 5))
status_counts = data['Status'].value_counts()
sns.barplot(x=status_counts.index, y=status_counts.values, palette='viridis')
plt.title('Number of Document Processing Tasks by Status')
plt.xlabel('Status')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Box plot of processing times by status
plt.figure(figsize=(10, 5))
sns.boxplot(x='Status', y='Processing Time (s)', data=data, palette='coolwarm')
plt.title('Distribution of Processing Times by Task Status')
plt.xlabel('Status')
plt.ylabel('Processing Time (s)')
plt.ylim(0, max(data['Processing Time (s)']) + 5)  # Set the y-limit for better visibility
plt.tight_layout()
plt.show()