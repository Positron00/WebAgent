# Performance Issue (Fixed Version)
# Query: Write a function that finds all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm.

def find_primes(limit):
    """
    Find all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm.
    
    Args:
        limit (int): Upper bound of the range to find primes in
        
    Returns:
        list: A list of prime numbers from 2 to limit
    """
    # Initialize the sieve array with True values
    sieve = [True] * (limit + 1)
    
    # 0 and 1 are not prime
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    
    # Apply the sieve algorithm
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            # Mark all multiples of i as non-prime
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    
    # Collect the prime numbers
    primes = [i for i in range(limit + 1) if sieve[i]]
    return primes

# Example usage
if __name__ == "__main__":
    # Test with different limits
    limits = [10, 100, 1000]
    for limit in limits:
        primes = find_primes(limit)
        print(f"Primes up to {limit}: {len(primes)}")
        if limit <= 100:  # Only print full list for smaller limits
            print(primes)