# Performance Issue
# Query: Write a function that finds all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm.

# WARNING: Using fallback example, actual code generation failed

def find_primes(limit):
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes