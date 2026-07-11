# This Python program implements the following use case:
# Write code to find BinaryGap of a given positive integer

def binary_gap(n):
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    
    binary_representation = bin(n)[2:]
    max_gap = 0
    current_gap = 0
    in_gap = False
    
    for digit in binary_representation:
        if digit == '1':
            if in_gap:
                max_gap = max(max_gap, current_gap)
            current_gap = 0
            in_gap = True
        elif in_gap:
            current_gap += 1
    
    return max_gap

# Example usage
examples = [9, 529, 20, 15, 32, 1041]
for number in examples:
    print(f"Binary gap of {number} is {binary_gap(number)}")