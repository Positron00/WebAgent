# Simple Data Processing Function
# Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_user_data(file_path='user_data.csv'):
    """
    Load user data from a CSV file. If the file does not exist,
    generate sample user data.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing user data.
    """
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}.")
    except FileNotFoundError:
        print(f"{file_path} not found. Generating sample data.")
        # Generate sample data
        sample_data = {
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'email': ['alice@example.com', 'bob@example.com', 
                      'charlie@example.com', 'david@example.com', 
                      'eve@example.com'],
            'age': [22, 17, 24, 15, 30]
        }
        df = pd.DataFrame(sample_data)
    return df

def filter_users_over_18(df):
    """
    Filter users who are over 18 years old.

    Args:
        df (DataFrame): The DataFrame containing user data.

    Returns:
        DataFrame: A filtered DataFrame containing users over 18.
    """
    return df[df['age'] > 18]

def visualize_age_distribution(original_df, filtered_df):
    """
    Visualize the age distribution of all users and those over 18.

    Args:
        original_df (DataFrame): The original DataFrame containing user data.
        filtered_df (DataFrame): The filtered DataFrame containing users over 18.
    """
    plt.figure(figsize=(12, 6))

    # Set up the subplots
    plt.subplot(1, 2, 1)
    sns.histplot(original_df['age'], bins=10, kde=True, color='skyblue')
    plt.title('Age Distribution of All Users')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(filtered_df['age'], bins=10, kde=True, color='salmon')
    plt.title('Age Distribution of Users Over 18')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    # Load user data
    user_data = load_user_data()
    
    # Filter users who are over 18
    users_over_18 = filter_users_over_18(user_data)
    
    # Visualize the age distribution
    visualize_age_distribution(user_data, users_over_18)

if __name__ == '__main__':
    main()