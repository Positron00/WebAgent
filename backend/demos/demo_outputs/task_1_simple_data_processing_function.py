# Simple Data Processing Function
# Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_users_over_18(csv_file):
    """
    Parses a CSV file to filter users over 18 years old.

    Parameters:
    csv_file (str): The path to the CSV file containing user data.

    Returns:
    pd.DataFrame: A DataFrame containing users over 18 years old.
    """
    try:
        # Load the CSV file into a DataFrame
        users_df = pd.read_csv(csv_file)

        # Check if necessary columns are present
        if not all(col in users_df.columns for col in ['name', 'email', 'age']):
            raise ValueError("CSV must contain 'name', 'email', and 'age' columns.")

        # Filter users who are over 18 years old
        users_over_18 = users_df[users_df['age'] > 18]

        return users_over_18

    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def visualize_user_data(users_df):
    """
    Visualizes the age distribution and the proportion of users over 18.

    Parameters:
    users_df (pd.DataFrame): A DataFrame containing user data.
    """
    if users_df.empty:
        print("No data to visualize.")
        return

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Create a histogram of ages
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(users_df['age'], bins=10, kde=True, color='blue')
    plt.title('Age Distribution of Users')
    plt.xlabel('Age')
    plt.ylabel('Number of Users')

    # Create a pie chart for the proportion of users over and under 18
    age_counts = {
        'Under 18': len(users_df[users_df['age'] <= 18]),
        '18 and Over': len(users_df[users_df['age'] > 18])
    }

    plt.subplot(1, 2, 2)
    plt.pie(age_counts.values(), labels=age_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['orange', 'lightgreen'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
    plt.title('Proportion of Users Over and Under 18')

    # Show the plots
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Assuming you have a CSV file named 'users.csv'
    users_df = parse_users_over_18('users.csv')
    visualize_user_data(users_df)