# Simple Data Processing with Bug (Fixed Version)
# Query: Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old. The function should handle errors gracefully.

# WARNING: Using fallback example, actual code review failed

import csv

def parse_user_data(file_path):
    adult_users = []
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    age = int(row.get('age', 0))
                    if age >= 18:  # Changed from > 18 to >= 18
                        adult_users.append(row)
                except (ValueError, TypeError) as e:
                    print(f"Invalid age for user {row.get('name', 'unknown')}: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
    return adult_users

if __name__ == "__main__":
    # Create a sample CSV file for testing
    sample_data = [
        {"name": "Alice", "email": "alice@example.com", "age": "25"},
        {"name": "Bob", "email": "bob@example.com", "age": "17"},
        {"name": "Charlie", "email": "charlie@example.com", "age": "30"},
        {"name": "David", "email": "david@example.com", "age": "invalid"},
    ]
    
    sample_file = "sample_users.csv"
    try:
        with open(sample_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["name", "email", "age"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_data)
        
        users = parse_user_data(sample_file)
        print(f"Found {len(users)} adult users")
        for user in users:
            print(f"Name: {user['name']}, Email: {user['email']}, Age: {user['age']}")
    finally:
        import os
        if os.path.exists(sample_file):
            os.remove(sample_file)