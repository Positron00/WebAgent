# Web API Endpoint
# Write a FastAPI endpoint that accepts file uploads, validates them for size and type (PDF only), and returns a JSON response with the file's metadata.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generate synthetic data for the visualization
# Simulating file metadata
np.random.seed(42)  # For reproducibility

# Simulated file sizes (in KB) for 100 uploads
file_sizes_kb = np.random.randint(1, 5000, size=100)  # Sizes between 1KB and 5000KB
file_types = np.random.choice(['pdf', 'docx', 'jpg', 'png', 'pdf', 'pdf'], size=100, p=[0.1, 0.1, 0.2, 0.2, 0.3, 0.1])

# Create a DataFrame to hold the simulated data
file_data = pd.DataFrame({
    'file_size_kb': file_sizes_kb,
    'file_type': file_types
})

# Set up the matplotlib figure
plt.figure(figsize=(14, 6))

# Plot 1: Histogram of file sizes
plt.subplot(1, 2, 1)
sns.histplot(file_data['file_size_kb'], bins=30, kde=True, color='blue', stat="density", alpha=0.6)
plt.title('Distribution of File Sizes')
plt.xlabel('File Size (KB)')
plt.ylabel('Density')
plt.grid(True)

# Plot 2: Count of file types
plt.subplot(1, 2, 2)
sns.countplot(x='file_type', data=file_data, palette='viridis')
plt.title('Count of Uploaded File Types')
plt.xlabel('File Type')
plt.ylabel('Count')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()