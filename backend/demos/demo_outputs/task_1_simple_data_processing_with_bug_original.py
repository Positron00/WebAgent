# Simple Data Processing with Bug
# Query: Write a Python function to parse a CSV file containing user data (name, email, age) and return a list of users who are over 18 years old. The function should handle errors gracefully.

# WARNING: Using fallback example, actual code generation failed

import csv

def parse_user_data(file_path):
    adult_users = []
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    age = int(row.get('age', 0))
                    if age > 18:
                        adult_users.append(row)
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
    return adult_users

if __name__ == "__main__":
    users = parse_user_data("users.csv")
    print(f"Found {len(users)} adult users")
    for user in users:
        print(f"Name: {user['name']}, Email: {user['email']}, Age: {user['age']}")