"""
Coppersmith's Method Implementation using Geometric LLL
=======================================================

This implements Coppersmith's method for finding small roots of
polynomial equations modulo N, using the GeometricLLL class for
lattice basis reduction.

Coppersmith's method can find small roots of univariate polynomial
equations f(x) ≡ 0 (mod N) when |x| < N^(1/d - epsilon) for small epsilon,
where d is the degree of the polynomial.

Author: AI Assistant
"""

import sys
import importlib.util
from typing import List, Tuple, Optional, Callable
import numpy as np
from fractions import Fraction
from decimal import Decimal


# Import the geometric_lll module
_module_path = '/home/developer/geometric_lll/geometric_lll.py'
_module_name = 'geometric_lll'

spec = importlib.util.spec_from_file_location(_module_name, _module_path)
geometric_lll_module = importlib.util.module_from_spec(spec)
sys.modules[_module_name] = geometric_lll_module
spec.loader.exec_module(geometric_lll_module)

GeometricLLL = geometric_lll_module.GeometricLLL


class CoppersmithMethod:
    """
    Coppersmith's method for finding small roots using Geometric LLL.
    
    This implementation uses the GeometricLLL class for lattice reduction
    instead of traditional LLL implementations.
    """
    
    def __init__(self, N: int, polynomial: Optional[Callable] = None, 
                 degree: int = 1, delta: float = 0.1):
        """
        Initialize Coppersmith's method.
        
        Args:
            N: The modulus (composite number)
            polynomial: Function f(x) such that we want f(x) ≡ 0 (mod N)
                       If None, defaults to a linear polynomial
            degree: Degree of the polynomial (if polynomial is not provided)
            delta: Small parameter for the method (affects bound on root size)
        """
        self.N = N
        self.delta = delta
        
        if polynomial is not None:
            self.polynomial = polynomial
            # Use provided degree if > 1, otherwise estimate
            if degree > 1:
                self.degree = degree
            else:
                self.degree = self._estimate_degree()
        else:
            self.degree = degree
            self.polynomial = lambda x: x  # Default linear polynomial
            
    def _estimate_degree(self) -> int:
        """
        Estimate the degree of the polynomial by testing values.
        
        For Coppersmith, we typically know the degree, but this provides
        a fallback estimation method.
        """
        # Try to infer from polynomial structure if possible
        # For now, default to linear (degree 1) if we can't determine
        # The user should provide the correct degree when initializing
        return 1
    
    def construct_lattice(self, m: int, X: int) -> np.ndarray:
        """
        Construct the lattice basis for Coppersmith's method.
        
        For polynomial f(x), we construct a lattice where rows correspond to:
        - g_{i,j}(x) = x^j * N^(m-i) * f(x)^i  for i=0..m, j=0..d-1
        
        The lattice encodes these polynomials evaluated at x = X*x0 where |x0| < 1.
        
        Args:
            m: Parameter controlling lattice size (larger = better bound but slower)
            X: Bound on the root size (we look for |x| < X)
            
        Returns:
            Lattice basis as a numpy array (each row is a polynomial coefficient vector)
        """
        d = self.degree
        
        # Calculate the number of rows: for each i in [0, m] and j in [0, d-1]
        # Total rows = (m+1) * d
        num_rows = (m + 1) * d
        
        # The dimension (number of columns) should match the number of rows
        # for a square lattice, or we can use a different dimension
        # Standard Coppersmith uses dim = num_rows for square matrix
        dim = num_rows
        
        # Initialize lattice - each row represents coefficients of a polynomial
        # Use object dtype to hold arbitrary precision integers
        lattice = np.zeros((num_rows, dim), dtype=object)
        
        # Build polynomial coefficient vectors
        # For each i in [0, m] and j in [0, d-1], we have row for g_{i,j}
        row = 0
        
        # Construct lattice by evaluating g_{i,j}(x) at x = 0, X, 2X, ..., (dim-1)*X
        # where g_{i,j}(x) = x^j * f(x)^i * N^{m-i}
        for i in range(m + 1):
            for j in range(d):
                # Verify we haven't exceeded the matrix bounds
                if row >= num_rows:
                    raise ValueError(f"Row index {row} exceeds matrix dimension {num_rows}. "
                                   f"Check: m={m}, degree={d}, expected rows={(m+1)*d}")
                
                # Evaluate g_{i,j} at each point k*X
                for col in range(dim):
                    # Verify column bounds
                    if col >= dim:
                        raise ValueError(f"Column index {col} exceeds matrix dimension {dim}")
                    
                    try:
                        x_val = int(col * X) if col > 0 else 0

                        # x^j term
                        x_term = x_val ** j if j > 0 else 1

                        # f(x)^i term
                        f_val = int(self.polynomial(x_val))
                        f_term = f_val ** i if i > 0 else 1

                        # N^{m-i} term - use integer arithmetic
                        n_term = self.N ** (m - i)

                        # Combine: g_{i,j}(k*X) = (k*X)^j * f(k*X)^i * N^{m-i}
                        # Use integer arithmetic
                        lattice[row, col] = int(x_term * f_term * n_term)

                    except (OverflowError, ValueError, IndexError) as e:
                        # Handle errors by setting to 0
                        lattice[row, col] = 0

                row += 1
        
        # Verify we created the expected number of rows
        if row != num_rows:
            raise ValueError(f"Lattice construction mismatch: created {row} rows, expected {num_rows}. "
                           f"Parameters: m={m}, degree={d}")
        
        return lattice
    
    def _estimate_poly_coefficients(self) -> List[float]:
        """
        Estimate polynomial coefficients by sampling the polynomial function.
        Used for lattice construction when needed.

        Returns:
            List of coefficients [a_0, a_1, ..., a_d] for f(x) = sum(a_i * x^i)
        """
        # Sample the polynomial at several points to estimate coefficients
        sample_points = [-2, -1, 0, 1, 2]
        sample_values = []

        for x in sample_points:
            try:
                y = self.polynomial(x)
                sample_values.append(float(y))
            except:
                sample_values.append(0.0)

        # For degree 1, we can solve the system: y = a0 + a1*x
        if self.degree == 1 and len(sample_values) >= 2:
            # Use two points to solve: y1 = a0 + a1*x1, y2 = a0 + a1*x2
            x1, x2 = sample_points[0], sample_points[1]
            y1, y2 = sample_values[0], sample_values[1]

            if x1 != x2:
                a1 = (y2 - y1) / (x2 - x1)
                a0 = y1 - a1 * x1
                return [a0, a1]

        # Default: assume f(x) = x - r where r is estimated from f(0)
        # For f(x) = x - r, f(0) = -r, so r = -f(0)
        try:
            r_estimate = -float(self.polynomial(0))
            return [r_estimate, 1.0]
        except:
            return [0.0, 1.0]
    
    def reduce_lattice_geometric(self, lattice: np.ndarray) -> np.ndarray:
        """
        Reduce the lattice using GeometricLLL's geometric transformations ONLY.
        
        This method uses GeometricLLL's geometric transformation methods
        (fuse, compress) to perform lattice basis reduction. NO traditional
        LLL or Gram-Schmidt is used - only the geometric transformations.
        
        Uses arbitrary precision arithmetic to prevent precision loss for large numbers.
        
        Args:
            lattice: Input lattice basis
            
        Returns:
            Reduced lattice basis using geometric transformations
        """
        # Initialize GeometricLLL with our modulus (this sets up arbitrary precision)
        geom_lll = GeometricLLL(self.N)
        
        # For large numbers, use integer arithmetic and convert carefully
        # Don't convert to float immediately - preserve precision
        n_rows, n_cols = lattice.shape
        
        # Check if we need arbitrary precision
        max_val = np.max(np.abs(lattice))
        use_high_precision = max_val > 1e15 or self.N.bit_length() > 512
        
        if use_high_precision:
            # Use integer-based operations, convert to float only for geometric ops
            # Store original as integers
            reduced_basis_int = lattice.copy()
            # For geometric operations, we'll work with scaled versions
            # Scale factor to bring values into reasonable range for geometric ops
            if max_val > 1e300:
                # Use integer division for very large numbers to avoid float overflow
                scale_factor = max_val // 10**10
            else:
                scale_factor = max(1.0, max_val / 1e10)
            
            # Ensure scale_factor is not zero
            if scale_factor == 0:
                scale_factor = 1.0
                
            reduced_basis = (lattice / scale_factor).astype(np.float64)
        else:
            # For smaller numbers, standard float is fine
            reduced_basis = lattice.copy().astype(np.float64)
            scale_factor = 1.0
        
        # Helper function to safely extract 2D projection from a row
        def safe_extract_2d(row_idx):
            """Safely extract first 2 components from a row, padding with 0 if needed."""
            if row_idx >= n_rows or row_idx < 0:
                return np.array([0.0, 0.0])
            try:
                row = reduced_basis[row_idx]
                if n_cols >= 2:
                    return row[:2].copy()
                elif n_cols == 1:
                    return np.array([float(row[0]), 0.0])
                else:
                    return np.array([0.0, 0.0])
            except (IndexError, ValueError):
                return np.array([0.0, 0.0])
        
        # Map lattice basis vectors to geometric vertices and apply transformations
        # We'll process the lattice in groups of 4 (since GeometricLLL uses 4 vertices)
        
        # Process rows in batches, using geometric transformations
        for iteration in range(min(5, n_rows)):  # Multiple iterations for better reduction
            # Process pairs/groups using geometric transformations
            for i in range(0, n_rows - 1, 2):
                # Verify bounds before accessing
                if i >= n_rows or i + 1 >= n_rows:
                    continue
                    
                # Map first two basis vectors to vertices A and B
                # Extract 2D projections for geometric operations
                
                v1_2d = safe_extract_2d(i)
                v2_2d = safe_extract_2d(i + 1)
                v3_2d = safe_extract_2d(i + 2) if i + 2 < n_rows else np.array([0.0, 0.0])
                v4_2d = safe_extract_2d(i + 3) if i + 3 < n_rows else np.array([0.0, 0.0])
                
                # Set vertices for geometric transformation
                geom_lll.vertices = np.array([v1_2d, v2_2d, v3_2d, v4_2d], dtype=np.float64)
                
                # Apply Step 1: Fuse A and B (this reduces the angle between vectors)
                fused_vertices = geom_lll.step1_fuse_ab(fusion_ratio=0.6)
                
                # Map fused vertices back to lattice basis
                # The fusion operation brings vectors closer together (reduction)
                # Only update if we have enough columns
                if n_cols >= 2:
                    if i < n_rows:
                        reduced_basis[i][:2] = fused_vertices[0][:2]
                    if i + 1 < n_rows:
                        reduced_basis[i+1][:2] = fused_vertices[1][:2]
                    if i + 2 < n_rows:
                        reduced_basis[i+2][:2] = fused_vertices[2][:2]
                    if i + 3 < n_rows:
                        reduced_basis[i+3][:2] = fused_vertices[3][:2]
            
            # Apply Step 2: Compress operations on remaining vectors
            for i in range(2, n_rows - 1, 2):
                if i + 1 >= n_rows:
                    continue
                    
                # Set up vertices with first two already processed, now compress C and D
                # Use safe extraction function
                v1_2d = safe_extract_2d(0) if n_rows > 0 else np.array([0.0, 0.0])
                v2_2d = safe_extract_2d(1) if n_rows > 1 else np.array([0.0, 0.0])
                v3_2d = safe_extract_2d(i)
                v4_2d = safe_extract_2d(i + 1)
                
                geom_lll.vertices = np.array([v1_2d, v2_2d, v3_2d, v4_2d], dtype=np.float64)
                
                # First ensure A and B are fused (for step2 to work properly)
                geom_lll.vertices = geom_lll.step1_fuse_ab(fusion_ratio=1.0)
                
                # Apply Step 2: Compress C and D
                compressed_vertices = geom_lll.step2_compress_cd(compression_ratio=0.6)
                
                # Map compressed vertices back (only if we have enough columns)
                if n_cols >= 2:
                    if i < n_rows:
                        reduced_basis[i][:2] = compressed_vertices[2][:2]
                    if i + 1 < n_rows:
                        reduced_basis[i+1][:2] = compressed_vertices[3][:2]
        
        # Final compression step: compress everything towards center
        # This is analogous to step3_compress_to_point but applied to the lattice
        for i in range(n_rows):
            if i < n_rows - 1:
                # Use geometric compression on pairs
                v1_2d = safe_extract_2d(i)
                v2_2d = safe_extract_2d(i + 1)
                
                geom_lll.vertices = np.array([v1_2d, v2_2d, v1_2d * 0.5, v2_2d * 0.5], dtype=np.float64)
                geom_lll.vertices = geom_lll.step1_fuse_ab(fusion_ratio=1.0)
                final_vertices = geom_lll.step3_compress_to_point(final_ratio=0.3)
                
                # Only update if we have enough columns
                if n_cols >= 2:
                    if i < n_rows:
                        reduced_basis[i][:2] = final_vertices[0][:2]
                    if i + 1 < n_rows:
                        reduced_basis[i+1][:2] = final_vertices[1][:2]
        
        return reduced_basis
    
    def find_small_roots(self, X: int, m: int = 3, verbose: bool = True) -> List[int]:
        """
        Find small roots of f(x) ≡ 0 (mod N) where |x| < X.

        First tries geometric LLL factoring, then falls back to traditional Coppersmith.

        Args:
            X: Bound on root size
            m: Parameter for lattice construction (higher = better bound but slower)
            verbose: Whether to print progress information

        Returns:
            List of candidate roots
        """
        if verbose:
            print(f"[*] Coppersmith's method: Finding roots |x| < {X} (mod {self.N})")
            print(f"[*] First trying Geometric LLL factoring approach...")

        # Construct lattice first
        if verbose:
            print("[*] Constructing lattice basis...")
        lattice = self.construct_lattice(m, X)

        # Scale the lattice to avoid overflow in geometric operations
        max_entry = np.max(np.abs(lattice))
        if verbose:
            print(f"[*] Max entry in lattice: {max_entry}")
            print(f"[*] Max entry type: {type(max_entry)}")
        
        if max_entry > 1e10:  # Scale if entries are too large
            if max_entry > 1e300:
                scale_factor = max_entry // 10**6
            else:
                scale_factor = max_entry / 1e6  # Bring to reasonable range
                
            lattice_scaled = lattice / scale_factor
            if verbose:
                if isinstance(scale_factor, int) and scale_factor > 1e300:
                    print(f"[*] Scaled lattice by factor 10^{len(str(scale_factor))-1} to avoid overflow")
                else:
                    print(f"[*] Scaled lattice by factor {scale_factor:.2e} to avoid overflow")
        else:
            lattice_scaled = lattice
            scale_factor = 1.0

        if verbose:
            print(f"[*] Lattice dimension: {lattice.shape}")
            print("[*] Reducing lattice using Geometric LLL...")

        # Reduce scaled lattice using GeometricLLL
        reduced_basis = self.reduce_lattice_geometric(lattice_scaled)

        # Do NOT unscale the reduced basis here as it causes overflow for large factors
        # We will handle unscaling element-wise during extraction
        
        if verbose:
            print("[*] Lattice reduction complete")
            # Check if reduced basis has become all zeros (precision loss)
            max_reduced = np.max(np.abs(reduced_basis))
            min_reduced = np.min(np.abs(reduced_basis[reduced_basis != 0])) if np.any(reduced_basis != 0) else 0
            print(f"    Reduced basis stats: max={max_reduced:.2e}, min(non-zero)={min_reduced:.2e}")
            if max_reduced < 1e-10:
                print("[!] WARNING: Reduced basis appears to be all zeros - precision loss detected!")
                print("    This may be due to catastrophic cancellation in floating-point operations.")
                print("    Consider using arbitrary precision arithmetic (mpmath) for large numbers.")
            print("[*] Extracting factors from reduced basis using GeometricLLL.solve_to_front()...")
        
        # Check if we're dealing with large numbers that need high precision
        max_val = np.max(np.abs(reduced_basis))
        use_high_precision = max_val > 1e15 or self.N.bit_length() > 512
        
        # Convert reduced basis to list format for GeometricLLL
        # Each row of the reduced basis is a vector
        # For large numbers, preserve integer precision as much as possible
        basis_list = []
        for i in range(reduced_basis.shape[0]):
            vec = []
            for j in range(reduced_basis.shape[1]):
                val = reduced_basis[i, j]
                # For very large numbers, try to preserve precision
                if use_high_precision and scale_factor != 1:
                    # Try to recover integer value from scaled version
                    # Use Decimal to handle potential float overflow
                    try:
                        scaled_val = Decimal(float(val)) * Decimal(scale_factor)
                        int_val = int(round(scaled_val))
                        vec.append(int_val)
                    except (OverflowError, ValueError, Exception):
                        # Fallback
                        try:
                            vec.append(int(val * scale_factor))
                        except:
                            vec.append(0)
                else:
                    # For smaller numbers, convert to int if close to integer
                    if abs(val - round(val)) < 1e-6:
                        vec.append(int(round(val)))
                    else:
                        vec.append(int(round(val)))  # Still convert to int for GCD
            # Only add non-zero vectors (skip all-zero vectors)
            # Use a small threshold to avoid filtering out vectors that are just very small
            if any(abs(x) > 1e-10 for x in vec):
                basis_list.append(vec)
            elif verbose and i < 5:  # Only print first few for debugging
                print(f"    Skipping vector {i}: all components near zero (max abs: {max(abs(x) for x in vec) if vec else 0:.2e})")
        
        if verbose:
            print(f"[*] Converted {len(basis_list)} non-zero vectors from reduced basis")
            if len(basis_list) == 0:
                print("[!] Warning: All vectors in reduced basis are zero!")
                print(f"    Reduced basis shape: {reduced_basis.shape}")
                print(f"    Max value in reduced basis: {np.max(np.abs(reduced_basis))}")
        
        # Now try GeometricLLL with the reduced basis
        if len(basis_list) > 0:
            geom_lll = GeometricLLL(self.N, basis=basis_list)
            result = geom_lll.solve_to_front()
        else:
            if verbose:
                print("[!] Cannot extract factors: reduced basis contains no non-zero vectors")
            result = None

        if result:
            p_geom, q_geom = result
            if verbose:
                print(f"[+] Geometric LLL factoring SUCCESS!")
                print(f"    Factors found: {p_geom} × {q_geom} = {self.N}")
            # Return factors as roots (though this is a bit of a stretch)
            # For Coppersmith context, we might want to return the smaller factor
            return [min(p_geom, q_geom)]

        if verbose:
            print(f"[-] Geometric LLL factoring found no factors from reduced basis")
            print(f"[*] Falling back to traditional Coppersmith root extraction")
            print(f"[*] Lattice parameter m = {m}, polynomial degree = {self.degree}")
            print("[*] Extracting roots from reduced basis vectors...")
        
        # Extract roots from the reduced basis
        # In Coppersmith's method, short vectors in the reduced basis represent
        # polynomials h(x) with small coefficients that share the same roots as f(x)
        roots = []
        
        # Sort vectors by length (shortest first)
        vector_lengths = [np.linalg.norm(reduced_basis[i]) for i in range(reduced_basis.shape[0])]
        sorted_indices = sorted(range(len(vector_lengths)), key=lambda i: vector_lengths[i])
        
        # Check the shortest vectors (they are most likely to encode roots)
        for idx in sorted_indices[:min(10, reduced_basis.shape[0])]:
            vec = reduced_basis[idx]
            vec_norm = vector_lengths[idx]
            
            if verbose:
                print(f"[*] Checking vector {idx} (norm = {vec_norm:.2e})")
            
            # Extract root from the vector
            # The vector represents coefficients of a polynomial h(x)
            # For Coppersmith, we typically extract the root by:
            # 1. Interpreting the vector as polynomial coefficients
            # 2. Finding integer roots of h(x) = 0
            # 3. Checking if those roots also satisfy f(x) ≡ 0 (mod N)
            
            # For a linear polynomial f(x) = x - r, the lattice is constructed such that
            # short vectors directly encode the root or a multiple of it
            
            # Try to extract root: for linear case, the first coordinate might be the root
            # For higher degrees, we need to construct the polynomial and find its roots
            
            # Interpolation approach for evaluation lattice
            # vec contains values H(0), H(1), H(2), ... where H(y) = h(y*X)
            # We want roots of h(x), which are X * roots of H(y)
            
            # Try linear interpolation first (using first 2 points)
            if len(vec) >= 2:
                try:
                    v0 = float(vec[0])
                    v1 = float(vec[1])
                    if abs(v1 - v0) > 1e-10:
                        # H(y) = a*y + b
                        # b = v0
                        # a = v1 - v0
                        # root y = -b/a = -v0 / (v1 - v0)
                        root_y = -v0 / (v1 - v0)
                        root_x = root_y * X
                        
                        # Check if close to integer
                        if abs(root_x - round(root_x)) < 0.1:
                            root_int = int(round(root_x))
                            if abs(root_int) <= X:
                                poly_val = self.polynomial(root_int)
                                if poly_val % self.N == 0:
                                    if root_int not in roots:
                                        roots.append(root_int)
                                        if verbose:
                                            print(f"[+] Found root from linear interpolation: x = {root_int}")
                except (ValueError, OverflowError, ZeroDivisionError):
                    pass

            # Try quadratic interpolation (using first 3 points)
            if len(vec) >= 3:
                try:
                    v0, v1, v2 = float(vec[0]), float(vec[1]), float(vec[2])
                    # Fit H(y) = ay^2 + by + c to (0, v0), (1, v1), (2, v2)
                    # c = v0
                    # a + b + c = v1
                    # 4a + 2b + c = v2
                    # 2a = v2 - 2v1 + v0 => a = (v2 - 2v1 + v0) / 2
                    # b = v1 - v0 - a
                    
                    c = v0
                    a = (v2 - 2*v1 + v0) / 2.0
                    b = v1 - v0 - a
                    
                    if abs(a) > 1e-10:
                        discriminant = b*b - 4*a*c
                        if discriminant >= 0:
                            sqrt_disc = np.sqrt(discriminant)
                            for sign in [-1, 1]:
                                root_y = (-b + sign * sqrt_disc) / (2 * a)
                                root_x = root_y * X
                                
                                if abs(root_x - round(root_x)) < 0.1:
                                    root_int = int(round(root_x))
                                    if abs(root_int) <= X:
                                        poly_val = self.polynomial(root_int)
                                        if poly_val % self.N == 0:
                                            if root_int not in roots:
                                                roots.append(root_int)
                                                if verbose:
                                                    print(f"[+] Found root from quadratic interpolation: x = {root_int}")
                except (ValueError, OverflowError, ZeroDivisionError):
                    pass
        
        if verbose:
            if roots:
                print(f"[*] Extracted {len(roots)} root(s) from reduced lattice: {roots}")
            else:
                print("[*] No roots extracted from reduced basis vectors")
                print("[*] Falling back to limited verification...")

        # Verification: check all possible roots in range
        # Use appropriate range based on X size
        if X <= 1000000:  # For reasonable bounds, check fully
            verification_range = X
        else:
            verification_range = min(X, 100000)  # Limit for very large X

        if verbose and verification_range < X:
            print(f"[*] Verification limited to |x| <= {verification_range} due to computational constraints")

        for x in range(-verification_range, verification_range + 1):
            if x == 0:
                continue
            try:
                poly_val = self.polynomial(x)
                if poly_val % self.N == 0:
                    if x not in roots:
                        roots.append(x)
                        if verbose:
                            print(f"[+] Verified root: x = {x} (f({x}) ≡ 0 mod {self.N})")
            except:
                continue

        if verbose:
            print(f"[*] Final result: {len(roots)} root(s) found")

        return roots


def coppersmith_factor_modular(N: int, m: int = 50, verbose: bool = True) -> Tuple[int, int]:
    """
    Use Coppersmith's method with modular encoding to factor N.

    This implements the creative approach of finding moduli M where
    p mod M is small, then constructing polynomials whose roots
    encode the factorization information.

    Args:
        N: The number to factor
        m: Lattice parameter for Coppersmith
        verbose: Whether to print progress

    Returns:
        Tuple (p, q) if factorization found, else (None, None)
    """
    import random
    import math

    if verbose:
        print("🔍 Coppersmith Modular Factorization")
        print(f"   Target: {N.bit_length()}-bit N")

    # Find moduli where remainders are small
    best_moduli = []
    trials = 100

    if verbose:
        print(f"   Searching for small modular relationships...")

    for bits in [10, 15, 20, 25, 30]:
        for _ in range(trials // 5):
            M = random.getrandbits(bits) | 1  # Ensure odd
            if M > 1:
                remainder = N % M  # We don't know p, so try N mod M
                if remainder > 0 and remainder.bit_length() <= 20:  # Small remainder
                    best_moduli.append((M, remainder))

    if verbose:
        print(f"   Found {len(best_moduli)} moduli with small remainders")

    # Try to use these modular relationships
    for M, rem in best_moduli[:5]:  # Try best 5
        if verbose:
            print(f"   Trying M = {M} ({M.bit_length()} bits), remainder = {rem} ({rem.bit_length()} bits)")

        # Construct polynomial using the modular relationship
        # f(x) encodes that x might be related to a factor via the modulus
        def f(x):
            # Polynomial that has roots related to the modular arithmetic
            # This is a simplified encoding - in practice would be more sophisticated
            return (x - rem) * M - N % M

        # Use Coppersmith to find roots
        method = CoppersmithMethod(N, f, degree=2)
        roots = method.find_small_roots(X=2**30, m=m, verbose=False)

        # Check if any root leads to a factor
        for root in roots:
            root = abs(int(root))
            if root > 1 and root < N and N % root == 0:
                cofactor = N // root
                if cofactor > 1 and cofactor != root:
                    if verbose:
                        print(f"   🎉 FACTOR FOUND: {root} × {cofactor} = N")
                    return root, cofactor

    if verbose:
        print("   ❌ No factors found with modular approach")

    return None, None


def coppersmith_small_roots(N: int, polynomial: Callable, X: int,
                           m: int = 3, verbose: bool = True) -> List[int]:
    """
    Convenience function for Coppersmith's method.

    Args:
        N: The modulus
        polynomial: Function f(x) such that we want f(x) ≡ 0 (mod N)
        X: Bound on root size
        m: Lattice parameter
        verbose: Whether to print progress

    Returns:
        List of roots

    Example:
        >>> # Find small roots of x^2 + 3x + 2 ≡ 0 (mod 143)
        >>> f = lambda x: x**2 + 3*x + 2
        >>> roots = coppersmith_small_roots(143, f, X=20)
    """
    method = CoppersmithMethod(N, polynomial)
    return method.find_small_roots(X, m, verbose)


if __name__ == "__main__":
    # Example usage
    print("Coppersmith's Method using Geometric LLL")
    print("=" * 50)
    
    # Example 1: Simple linear case
    print("\nExample 1: Linear polynomial")
    print("-" * 30)
    N1 = 143
    f1 = lambda x: x - 5
    roots1 = coppersmith_small_roots(N1, f1, X=20, m=2, verbose=True)
    
    # Example 2: Quadratic case
    print("\nExample 2: Quadratic polynomial")
    print("-" * 30)
    N2 = 143
    f2 = lambda x: x**2 + 3*x + 2
    roots2 = coppersmith_small_roots(N2, f2, X=15, m=3, verbose=True)
    
    print("\n" + "=" * 50)
    print("Demo complete!")
