import csv

def parse_users_over_18(csv_file_path):
    '''
    Parse a CSV file containing user data and return users over 18.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries containing user data for users over 18
    '''
    users_over_18 = []
    
    try:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    age = int(row.get('age', 0))
                    if age > 18:
                        users_over_18.append(row)
                except (ValueError, TypeError):
                    # Skip rows with invalid age values
                    continue
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        
    return users_over_18