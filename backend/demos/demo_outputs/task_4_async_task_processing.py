# Async Task Processing
# Implement an asynchronous task processing system using Python's asyncio that can handle multiple document processing tasks in parallel with proper error handling and status updates.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the style for seaborn
sns.set(style="whitegrid")

# Generate a simulated dataset
def generate_simulated_data(num_tasks=100):
    """
    Generate a simulated dataset for task processing.
    
    Parameters:
    num_tasks (int): Number of tasks to simulate.
    
    Returns:
    pd.DataFrame: DataFrame containing task processing results.
    """
    np.random.seed(0)  # For reproducibility
    task_ids = np.arange(num_tasks)
    processing_time = np.random.exponential(scale=2, size=num_tasks)  # Simulated processing times
    success_rate = np.random.choice([True, False], size=num_tasks, p=[0.8, 0.2])  # 80% success rate

    return pd.DataFrame({
        'Task ID': task_ids,
        'Processing Time (s)': processing_time,
        'Success': success_rate
    })

# Create the dataset
tasks_df = generate_simulated_data()

# Data for visualizations
tasks_df['Status'] = np.where(tasks_df['Success'], 'Success', 'Failure')
tasks_over_time = tasks_df.groupby('Task ID').agg({'Processing Time (s)': 'sum', 'Success': 'mean'}).reset_index()
tasks_over_time['Success'] = tasks_over_time['Success'].apply(lambda x: 'Success' if x >= 0.8 else 'Failure')

# Set up the subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 1. Line plot of tasks processed over time
axs[0].plot(tasks_over_time['Task ID'], tasks_over_time['Processing Time (s)'], marker='o', color='b')
axs[0].set_title('Tasks Processed Over Time', fontsize=16)
axs[0].set_xlabel('Task ID', fontsize=14)
axs[0].set_ylabel('Processing Time (seconds)', fontsize=14)
axs[0].grid(True)

# 2. Bar chart for success and failure rates
sns.countplot(data=tasks_df, x='Status', ax=axs[1], palette='Set2')
axs[1].set_title('Success vs Failure of Tasks', fontsize=16)
axs[1].set_xlabel('Task Status', fontsize=14)
axs[1].set_ylabel('Number of Tasks', fontsize=14)

# 3. Pie chart for the proportion of successful vs failed tasks
task_counts = tasks_df['Status'].value_counts()
axs[2].pie(task_counts, labels=task_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
axs[2].axis('equal')  # Equal aspect ratio ensures that pie chart is circular
axs[2].set_title('Proportion of Successful vs Failed Tasks', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.show()