# Simple Data Processing Function
# Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_user_data(file_path):
    """
    Parses a CSV file to extract user data and return users over 18 years old.
    
    Parameters:
    - file_path: str, path to the CSV file containing user data.
    
    Returns:
    - List of users over 18 years old.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Check for necessary columns
        if not all(col in df.columns for col in ['name', 'email', 'age']):
            raise ValueError("CSV file must contain 'name', 'email', and 'age' columns.")
        
        # Filter users who are over 18 years old
        users_over_18 = df[df['age'] > 18]
        
        return users_over_18
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

def visualize_age_distribution(users_df):
    """
    Visualizes the age distribution of users using a histogram and pie chart.
    
    Parameters:
    - users_df: DataFrame containing user data.
    """
    # Set up the aesthetics for the plots
    sns.set(style='whitegrid')

    # Create a histogram of ages
    plt.figure(figsize=(12, 6))
    sns.histplot(users_df['age'], bins=10, kde=True, color='skyblue')
    plt.title('Age Distribution of Users Over 18 Years Old')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.axvline(x=18, color='red', linestyle='--', label='Age 18')
    plt.legend()
    plt.show()

    # Create a pie chart of age groups
    age_group_counts = {
        'Under 18': len(df[df['age'] <= 18]),
        'Over 18': len(users_df)
    }

    plt.figure(figsize=(8, 8))
    plt.pie(age_group_counts.values(), labels=age_group_counts.keys(), autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightskyblue'])
    plt.title('Proportion of Users Under and Over 18 Years Old')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Example usage
if __name__ == '__main__':
    file_path = 'user_data.csv'  # Specify your CSV file path here
    users_df = parse_user_data(file_path)
    
    if users_df is not None and not users_df.empty:
        visualize_age_distribution(users_df)
    else:
        print("No users over 18 found or data could not be loaded.")