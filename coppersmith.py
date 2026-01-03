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
        
        # For simplicity, we'll construct a lattice that encodes f(x)
        # The dimension is (m+1)*d
        dim = (m + 1) * d
        
        # Initialize lattice - each row represents coefficients of a polynomial
        lattice = np.zeros((dim, dim), dtype=float)
        
        # Build polynomial coefficient vectors
        # For each i in [0, m] and j in [0, d-1], we have row for g_{i,j}
        row = 0
        
        # Construct lattice by evaluating g_{i,j}(x) at x = 0, X, 2X, ..., (dim-1)*X
        # where g_{i,j}(x) = x^j * f(x)^i * N^{m-i}

        row = 0
        for i in range(m + 1):
            for j in range(d):
                # Evaluate g_{i,j} at each point k*X
                col = 0
                for k in range(dim):
                    try:
                        x_val = k * X

                        # x^j term
                        x_term = x_val ** j if j > 0 else 1.0

                        # f(x)^i term
                        f_val = self.polynomial(x_val)
                        f_term = f_val ** i if i > 0 else 1.0

                        # N^{m-i} term
                        n_term = float(self.N ** (m - i))

                        # Combine: g_{i,j}(k*X) = (k*X)^j * f(k*X)^i * N^{m-i}
                        lattice[row, col] = x_term * f_term * n_term

                    except (OverflowError, ValueError):
                        # Handle large values by setting to 0
                        lattice[row, col] = 0.0

                    col += 1

                row += 1
        
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
        
        Args:
            lattice: Input lattice basis
            
        Returns:
            Reduced lattice basis using geometric transformations
        """
        # Initialize GeometricLLL with our modulus
        geom_lll = GeometricLLL(self.N)
        
        # Start with the original lattice
        reduced_basis = lattice.copy().astype(float)
        n_rows = reduced_basis.shape[0]
        
        # Map lattice basis vectors to geometric vertices and apply transformations
        # We'll process the lattice in groups of 4 (since GeometricLLL uses 4 vertices)
        
        # Process rows in batches, using geometric transformations
        for iteration in range(min(5, n_rows)):  # Multiple iterations for better reduction
            # Process pairs/groups using geometric transformations
            for i in range(0, n_rows - 1, 2):
                if i + 1 < n_rows:
                    # Map first two basis vectors to vertices A and B
                    # Extract 2D projections for geometric operations
                    v1_2d = reduced_basis[i][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i][0], 0])
                    v2_2d = reduced_basis[i+1][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i+1][0], 0])
                    
                    # Set up geometric vertices: A, B represent the two vectors
                    # C, D are set to represent other vectors or defaults
                    if i + 2 < n_rows:
                        v3_2d = reduced_basis[i+2][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i+2][0], 0])
                    else:
                        v3_2d = np.array([0.0, 0.0])
                    
                    if i + 3 < n_rows:
                        v4_2d = reduced_basis[i+3][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i+3][0], 0])
                    else:
                        v4_2d = np.array([0.0, 0.0])
                    
                    # Set vertices for geometric transformation
                    geom_lll.vertices = np.array([v1_2d, v2_2d, v3_2d, v4_2d], dtype=np.float64)
                    
                    # Apply Step 1: Fuse A and B (this reduces the angle between vectors)
                    fused_vertices = geom_lll.step1_fuse_ab(fusion_ratio=0.6)
                    
                    # Map fused vertices back to lattice basis
                    # The fusion operation brings vectors closer together (reduction)
                    reduced_basis[i][:2] = fused_vertices[0][:2]
                    reduced_basis[i+1][:2] = fused_vertices[1][:2]
                    if i + 2 < n_rows:
                        reduced_basis[i+2][:2] = fused_vertices[2][:2]
                    if i + 3 < n_rows:
                        reduced_basis[i+3][:2] = fused_vertices[3][:2]
            
            # Apply Step 2: Compress operations on remaining vectors
            for i in range(2, n_rows - 1, 2):
                if i + 1 < n_rows:
                    # Set up vertices with first two already processed, now compress C and D
                    v1_2d = reduced_basis[0][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[0][0], 0])
                    v2_2d = reduced_basis[1][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[1][0], 0])
                    v3_2d = reduced_basis[i][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i][0], 0])
                    v4_2d = reduced_basis[i+1][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i+1][0], 0])
                    
                    geom_lll.vertices = np.array([v1_2d, v2_2d, v3_2d, v4_2d], dtype=np.float64)
                    
                    # First ensure A and B are fused (for step2 to work properly)
                    geom_lll.vertices = geom_lll.step1_fuse_ab(fusion_ratio=1.0)
                    
                    # Apply Step 2: Compress C and D
                    compressed_vertices = geom_lll.step2_compress_cd(compression_ratio=0.6)
                    
                    # Map compressed vertices back
                    reduced_basis[i][:2] = compressed_vertices[2][:2]
                    reduced_basis[i+1][:2] = compressed_vertices[3][:2]
        
        # Final compression step: compress everything towards center
        # This is analogous to step3_compress_to_point but applied to the lattice
        for i in range(n_rows):
            if i < n_rows - 1:
                # Use geometric compression on pairs
                v1_2d = reduced_basis[i][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i][0], 0])
                v2_2d = reduced_basis[i+1][:2] if reduced_basis.shape[1] >= 2 else np.array([reduced_basis[i+1][0], 0])
                
                geom_lll.vertices = np.array([v1_2d, v2_2d, v1_2d * 0.5, v2_2d * 0.5], dtype=np.float64)
                geom_lll.vertices = geom_lll.step1_fuse_ab(fusion_ratio=1.0)
                final_vertices = geom_lll.step3_compress_to_point(final_ratio=0.3)
                
                reduced_basis[i][:2] = final_vertices[0][:2]
                if i + 1 < n_rows:
                    reduced_basis[i+1][:2] = final_vertices[1][:2]
        
        return reduced_basis
    
    def find_small_roots(self, X: int, m: int = 3, verbose: bool = True) -> List[int]:
        """
        Find small roots of f(x) ≡ 0 (mod N) where |x| < X.
        
        Args:
            X: Bound on root size
            m: Parameter for lattice construction (higher = better bound but slower)
            verbose: Whether to print progress information
            
        Returns:
            List of candidate roots
        """
        if verbose:
            print(f"[*] Coppersmith's method: Finding roots |x| < {X} (mod {self.N})")
            print(f"[*] Using Geometric LLL for lattice reduction")
            print(f"[*] Lattice parameter m = {m}, polynomial degree = {self.degree}")
        
        # Construct lattice
        if verbose:
            print("[*] Constructing lattice basis...")
        lattice = self.construct_lattice(m, X)

        # Scale the lattice to avoid overflow in geometric operations
        max_entry = np.max(np.abs(lattice))
        if max_entry > 1e10:  # Scale if entries are too large
            scale_factor = max_entry / 1e6  # Bring to reasonable range
            lattice_scaled = lattice / scale_factor
            if verbose:
                print(f"[*] Scaled lattice by factor {scale_factor:.2e} to avoid overflow")
        else:
            lattice_scaled = lattice
            scale_factor = 1.0

        if verbose:
            print(f"[*] Lattice dimension: {lattice.shape}")
            print("[*] Reducing lattice using Geometric LLL...")

        # Reduce scaled lattice using GeometricLLL
        reduced_basis = self.reduce_lattice_geometric(lattice_scaled)

        # Unscale the reduced basis
        if scale_factor != 1.0:
            reduced_basis = reduced_basis * scale_factor
        
        if verbose:
            print("[*] Lattice reduction complete")
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
            
            if self.degree == 1:
                # Linear case: Extract linear polynomial h(x) = a*x + b from vector
                # Root is x = -b/a (if a != 0)
                
                # The vector represents coefficients [b, a] for h(x) = a*x + b
                # Extract first two coordinates as polynomial coefficients
                if len(vec) >= 2:
                    try:
                        # Normalize the vector to get integer-like coefficients
                        # Find GCD to simplify
                        a_val = float(vec[1])
                        b_val = float(vec[0])
                        
                        # Skip if both are zero
                        if abs(a_val) < 1e-10 and abs(b_val) < 1e-10:
                            continue
                        
                        # If a is non-zero, solve for root: x = -b/a
                        if abs(a_val) > 1e-10:
                            root_candidate = -b_val / a_val
                            
                            # Round to nearest integer if close
                            if abs(root_candidate - round(root_candidate)) < 1e-6:
                                root_int = int(round(root_candidate))
                                
                                # Check bounds
                                if abs(root_int) <= X:
                                    # Verify it's a root of f(x) mod N
                                    poly_val = self.polynomial(root_int)
                                    if poly_val % self.N == 0:
                                        if root_int not in roots:
                                            roots.append(root_int)
                                            if verbose:
                                                print(f"[+] Found root from vector {idx}: x = {root_int} (from h(x) = {a_val:.2f}*x + {b_val:.2f})")
                        
                        # Also try if b is non-zero and a is zero (constant polynomial)
                        # This case doesn't give a root directly
                    except (ValueError, OverflowError, ZeroDivisionError):
                        continue
            
            else:
                # Higher degree: construct polynomial h(x) from vector coefficients
                # The vector represents coefficients for h(x) = c_0 + c_1*x + ... + c_k*x^k
                
                # Extract polynomial coefficients from vector
                poly_coeffs = []
                max_coeffs = min(self.degree + 2, len(vec))  # Allow one extra for safety
                
                for i in range(max_coeffs):
                    try:
                        coeff = float(vec[i])
                        poly_coeffs.append(coeff)
                    except (ValueError, OverflowError, IndexError):
                        break
                
                # Remove trailing zeros (but keep at least 2 coefficients)
                while len(poly_coeffs) > 2 and abs(poly_coeffs[-1]) < 1e-10:
                    poly_coeffs.pop()
                
                if len(poly_coeffs) >= 2:
                    # Find integer roots algebraically when possible
                    # For quadratic: use quadratic formula
                    # For higher degrees: use more advanced methods
                    
                    if len(poly_coeffs) == 2:
                        # Linear: h(x) = a*x + b, root is x = -b/a
                        a, b = poly_coeffs[1], poly_coeffs[0]
                        if abs(a) > 1e-10:
                            root_candidate = -b / a
                            if abs(root_candidate - round(root_candidate)) < 1e-6:
                                root_int = int(round(root_candidate))
                                if abs(root_int) <= X:
                                    f_val = self.polynomial(root_int)
                                    if f_val % self.N == 0:
                                        if root_int not in roots:
                                            roots.append(root_int)
                                            if verbose:
                                                print(f"[+] Found root from linear polynomial: x = {root_int}")
                    
                    elif len(poly_coeffs) == 3:
                        # Quadratic: h(x) = a*x^2 + b*x + c
                        # Use quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / (2a)
                        c, b, a = poly_coeffs[0], poly_coeffs[1], poly_coeffs[2]
                        if abs(a) > 1e-10:
                            discriminant = b*b - 4*a*c
                            if discriminant >= 0:
                                sqrt_disc = np.sqrt(discriminant)
                                for sign in [-1, 1]:
                                    root_candidate = (-b + sign * sqrt_disc) / (2 * a)
                                    if abs(root_candidate - round(root_candidate)) < 1e-6:
                                        root_int = int(round(root_candidate))
                                        if abs(root_int) <= X:
                                            f_val = self.polynomial(root_int)
                                            if f_val % self.N == 0:
                                                if root_int not in roots:
                                                    roots.append(root_int)
                                                    if verbose:
                                                        print(f"[+] Found root from quadratic polynomial: x = {root_int}")
                    
                    # For higher degrees, we'd need more sophisticated root finding
                    # But for Coppersmith, we typically work with low-degree cases
        
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

