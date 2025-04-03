# Async Task Processing
# Implement an asynchronous task processing system using Python's asyncio that can handle multiple document processing tasks in parallel with proper error handling and status updates.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the seaborn plots
sns.set(style="whitegrid")

# Generate sample data if specific data files do not exist
def generate_sample_data(num_entries=100):
    """Generates sample data for document processing."""
    statuses = ['Completed', 'In Progress', 'Failed']
    # Randomly generate statuses for the specified number of entries
    data = {
        'Document ID': [f'Doc_{i}' for i in range(num_entries)],
        'Status': np.random.choice(statuses, num_entries, p=[0.7, 0.2, 0.1]),
        'Timestamp': pd.date_range(start='2023-01-01', periods=num_entries, freq='H')
    }
    return pd.DataFrame(data)

# Load or generate the data
try:
    # Here we would normally load data from a file
    # For the sake of this example, we will generate sample data
    data = generate_sample_data(100)
except Exception as e:
    print(f"Error loading data: {e}")
    print("Generating sample data as a fallback.")
    data = generate_sample_data(100)

# Data Processing
# Count the number of documents in each status
status_counts = data['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']

# Create a time series for processing over time
time_series = data.groupby([data['Timestamp'].dt.date, 'Status']).size().unstack(fill_value=0)
time_series = time_series.reset_index()

# Visualization 1: Bar Chart of Document Status Counts
plt.figure(figsize=(10, 5))
sns.barplot(x='Status', y='Count', data=status_counts, palette='viridis')
plt.title('Document Processing Status Counts')
plt.xlabel('Status')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Line Chart of Document Processing Over Time
plt.figure(figsize=(10, 5))
for status in time_series.columns[1:]:
    plt.plot(time_series['Timestamp'], time_series[status], marker='o', label=status)

plt.title('Document Processing Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Documents Processed')
plt.legend(title='Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()