#!/usr/bin/env python3
"""
Extract Modular Relationships for Coppersmith Attack
===================================================

This script extracts modular relationships from a given N that can be used
in Coppersmith's method for factorization. It finds moduli M where N mod M
has small remainders, which provide strong constraints for lattice construction.

The output is saved in a format compatible with serious_coppersmith_assault.py
"""

import math
import pickle
import random
import sys
from typing import List, Tuple
from collections import defaultdict


def extract_modular_relationships(N: int, max_relationships: int = 5000, 
                                   max_mod_bits: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    Extract modular relationships from N.
    
    A modular relationship is (M, remainder) where:
    - M is a modulus
    - remainder = N % M
    - Small remainders indicate strong constraints
    
    Args:
        N: The number to factor
        max_relationships: Maximum number of relationships to find
        max_mod_bits: Maximum bit length for moduli to test
        
    Returns:
        List of tuples: (M, remainder, remainder_bits, mod_bits)
        Sorted by remainder size (smallest first = strongest constraint)
    """
    print(f"[*] Extracting modular relationships for {N.bit_length()}-bit N...")
    print(f"[*] Searching for moduli up to {max_mod_bits} bits...")
    
    relationships = []
    N_bits = N.bit_length()
    
    # Strategy 1: Test powers of small primes
    print("[*] Strategy 1: Testing powers of small primes...")
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    for prime in small_primes:
        for power in range(1, max_mod_bits // prime.bit_length() + 1):
            M = prime ** power
            if M.bit_length() > max_mod_bits:
                break
            rem = N % M
            rem_bits = rem.bit_length()
            mod_bits = M.bit_length()
            
            # Keep if remainder is relatively small compared to modulus
            if rem_bits < mod_bits - 5:  # Remainder is at least 32x smaller
                relationships.append((M, rem, rem_bits, mod_bits))
    
    print(f"    Found {len(relationships)} relationships from prime powers")
    
    # Strategy 2: Test products of small primes (smooth numbers)
    print("[*] Strategy 2: Testing products of small primes (smooth numbers)...")
    smooth_products = []
    
    # Generate smooth numbers (products of small primes)
    def generate_smooth(max_bits, primes, current=1, start_idx=0):
        if current.bit_length() > max_bits:
            return
        if current > 1:
            smooth_products.append(current)
        for i in range(start_idx, len(primes)):
            new_current = current * primes[i]
            if new_current.bit_length() <= max_bits:
                generate_smooth(max_bits, primes, new_current, i)
    
    generate_smooth(max_mod_bits, small_primes[:10])  # Use first 10 primes
    
    for M in smooth_products:
        if M > 1:
            rem = N % M
            rem_bits = rem.bit_length()
            mod_bits = M.bit_length()
            
            if rem_bits < mod_bits - 3:  # Slightly more lenient for smooth numbers
                relationships.append((M, rem, rem_bits, mod_bits))
    
    print(f"    Found {len(relationships)} total relationships (including smooth numbers)")
    
    # Strategy 3: Random sampling of moduli in various bit ranges
    print("[*] Strategy 3: Random sampling of moduli...")
    samples_per_range = max_relationships // 10
    
    for target_bits in range(10, max_mod_bits + 1, 5):
        for _ in range(samples_per_range):
            # Generate random odd number in target bit range
            M = random.getrandbits(target_bits) | 1  # Ensure odd
            if M < 2:
                continue
            
            rem = N % M
            rem_bits = rem.bit_length()
            mod_bits = M.bit_length()
            
            # Keep if remainder is small relative to modulus
            # For random moduli, be more selective
            if rem_bits < mod_bits - 10:  # Remainder must be much smaller
                relationships.append((M, rem, rem_bits, mod_bits))
    
    print(f"    Found {len(relationships)} total relationships (including random samples)")
    
    # Strategy 4: Test moduli near powers of 2 (common in RSA)
    print("[*] Strategy 4: Testing moduli near powers of 2...")
    for exp in range(8, max_mod_bits):
        base = 2 ** exp
        # Test base ± small offsets
        for offset in [-3, -2, -1, 1, 2, 3, 5, 7, 9, 11]:
            M = base + offset
            if M < 2 or M.bit_length() > max_mod_bits:
                continue
            
            rem = N % M
            rem_bits = rem.bit_length()
            mod_bits = M.bit_length()
            
            if rem_bits < mod_bits - 5:
                relationships.append((M, rem, rem_bits, mod_bits))
    
    print(f"    Found {len(relationships)} total relationships (including power-of-2 neighbors)")
    
    # Strategy 5: Test moduli based on estimated factor sizes
    print("[*] Strategy 5: Testing moduli based on estimated factor sizes...")
    estimated_factor_bits = N_bits // 2
    
    # Test moduli that might reveal information about factors
    for factor_bits in range(max(8, estimated_factor_bits - 20), 
                            min(max_mod_bits, estimated_factor_bits + 20), 5):
        for _ in range(50):  # Sample 50 moduli per bit range
            M = random.getrandbits(factor_bits) | 1
            if M < 2:
                continue
            
            rem = N % M
            rem_bits = rem.bit_length()
            mod_bits = M.bit_length()
            
            if rem_bits < mod_bits - 8:
                relationships.append((M, rem, rem_bits, mod_bits))
    
    print(f"    Found {len(relationships)} total relationships (including factor-size-based)")
    
    # Remove duplicates (same M)
    seen_moduli = set()
    unique_relationships = []
    for rel in relationships:
        M = rel[0]
        if M not in seen_moduli:
            seen_moduli.add(M)
            unique_relationships.append(rel)
    
    print(f"[*] Removed duplicates: {len(unique_relationships)} unique relationships")
    
    # Sort by remainder size (smallest first = strongest constraint)
    # Secondary sort by remainder bit length
    unique_relationships.sort(key=lambda x: (x[1], x[2]))  # Sort by remainder value, then bits
    
    # Limit to top relationships
    if len(unique_relationships) > max_relationships:
        unique_relationships = unique_relationships[:max_relationships]
        print(f"[*] Limited to top {max_relationships} strongest relationships")
    
    print(f"[*] Extraction complete: {len(unique_relationships)} relationships")
    if unique_relationships:
        best = unique_relationships[0]
        print(f"[*] Strongest constraint: {best[2]}-bit remainder with {best[3]}-bit modulus")
    
    return unique_relationships


def save_relationships(N: int, relationships: List[Tuple[int, int, int, int]], 
                      filename: str = 'maximum_modular_relationships.pkl'):
    """
    Save modular relationships to a pickle file compatible with serious_coppersmith_assault.py
    
    Args:
        N: The number being factored
        relationships: List of (M, remainder, remainder_bits, mod_bits) tuples
        filename: Output filename
    """
    data = {
        'N': N,
        'N_bits': N.bit_length(),
        'relationships': relationships,
        'num_relationships': len(relationships)
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n[+] Saved {len(relationships)} relationships to {filename}")
    print(f"    N: {N.bit_length()}-bit number")
    print(f"    Top 5 strongest constraints:")
    for i, (M, rem, rem_bits, mod_bits) in enumerate(relationships[:5]):
        print(f"      {i+1}. M={M} ({mod_bits} bits), remainder={rem} ({rem_bits} bits)")


def main():
    """Main execution function"""
    print("=" * 70)
    print("MODULAR RELATIONSHIP EXTRACTION FOR COPPERSMITH ATTACK")
    print("=" * 70)
    print()
    
    # Get N from command line or prompt
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            print(f"❌ Error: '{sys.argv[1]}' is not a valid integer")
            sys.exit(1)
    else:
        # Prompt for N
        print("Enter the number N to extract modular relationships from:")
        print("(Or provide as command-line argument: python extract_modular_relationships.py <N>)")
        try:
            N_str = input("N: ").strip()
            N = int(N_str)
        except (ValueError, EOFError):
            print("❌ Error: Invalid input")
            sys.exit(1)
    
    if N < 2:
        print("❌ Error: N must be at least 2")
        sys.exit(1)
    
    print(f"\n[*] Target N: {N.bit_length()}-bit number")
    print(f"    N = {N}")
    print()
    
    # Extract relationships
    relationships = extract_modular_relationships(N, max_relationships=5000, max_mod_bits=50)
    
    if not relationships:
        print("\n❌ No strong modular relationships found!")
        print("   Try increasing max_mod_bits or checking if N has special structure")
        sys.exit(1)
    
    # Save to file
    save_relationships(N, relationships)
    
    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\n💡 Next steps:")
    print(f"   1. Run serious_coppersmith_assault.py to use these relationships")
    print(f"   2. The relationships will be used to construct the polynomial")
    print(f"   3. Stronger relationships (smaller remainders) are used first")
    print()


if __name__ == "__main__":
    main()
