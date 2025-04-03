# Simple Data Processing Function
# Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_users_over_18(file_path):
    """
    Parse the CSV file and return a list of users who are over 18 years old.
    
    Parameters:
    file_path (str): The path to the CSV file containing user data.
    
    Returns:
    list: A list of dictionaries of users over 18.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Check if required columns are in the DataFrame
        if not {'name', 'email', 'age'}.issubset(df.columns):
            raise ValueError("CSV must contain 'name', 'email', and 'age' columns.")

        # Filter users who are older than 18
        users_over_18 = df[df['age'] > 18]
        
        # Convert the filtered DataFrame to a list of dictionaries
        return users_over_18.to_dict(orient='records')
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def visualize_age_distribution(df):
    """
    Visualize the age distribution of users using a histogram and a bar plot.
    
    Parameters:
    df (DataFrame): The DataFrame containing user data.
    """
    plt.figure(figsize=(14, 6))

    # Histogram of ages
    plt.subplot(1, 2, 1)
    sns.histplot(df['age'], bins=10, kde=True, color='skyblue')
    plt.title('Age Distribution of Users')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    # Count of users above and below 18
    age_counts = df['age'].apply(lambda x: 'Over 18' if x > 18 else '18 or Under').value_counts()
    plt.subplot(1, 2, 2)
    sns.barplot(x=age_counts.index, y=age_counts.values, palette='viridis')
    plt.title('Count of Users by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Users')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = 'user_data.csv'  # Replace with your CSV file path
    users_over_18 = parse_users_over_18(file_path)
    
    # Visualize only if there are users over 18
    if users_over_18:
        df = pd.DataFrame(users_over_18)
        visualize_age_distribution(df)
    else:
        print("No users over 18 found.")