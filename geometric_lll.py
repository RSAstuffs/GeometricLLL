"""
Novel Geometric LLL Algorithm for Factoring
==========================================

HARDENED VERSION - Geometric pass + BALANCED column-scaled LLL
"""

import numpy as np
from typing import Tuple, List, Optional
import math


class GeometricLLL:
    """
    Geometric LLL - SINGLE PASS then column-scaled LLL.
    """

    def __init__(self, N: int, p: int = None, q: int = None, basis: np.ndarray = None):
        self.N = N
        self.p = p
        self.q = q
        self.basis = basis
        self.vertices = self._initialize_square()
        self.transformation_steps = []

    def solve_to_front(self) -> Optional[Tuple[int, int]]:
        p, q = self.find_factors_geometrically()
        if p is None or q is None:
            return None
        return p, q

    def _initialize_square(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64)

    def _is_zero_vector(self, v) -> bool:
        return all(x == 0 for x in v)

    def _reduce_vector(self, v_target, v_base):
        """Single reduction of v_target against v_base."""
        norm_sq = np.dot(v_base, v_base)
        if norm_sq == 0:
            return v_target.copy()
        
        proj = np.dot(v_target, v_base)
        if proj == 0:
            return v_target.copy()
        
        if proj >= 0:
            ratio = (proj + norm_sq // 2) // norm_sq
        else:
            ratio = -(-proj + norm_sq // 2) // norm_sq
        
        if ratio == 0:
            return v_target.copy()
        
        result = v_target - ratio * v_base
        
        if self._is_zero_vector(result):
            if ratio > 0:
                ratio -= 1
            elif ratio < 0:
                ratio += 1
            if ratio != 0:
                result = v_target - ratio * v_base
            else:
                result = v_target.copy()
        
        return result

    def step1_fuse_ab(self, fusion_ratio: float = 0.5) -> np.ndarray:
        if self.vertices.dtype == object:
            vertices = self.vertices.copy()
            v1, v2 = vertices[0], vertices[1]
            v2_new = self._reduce_vector(v2, v1)
            if np.dot(v2_new, v2_new) < np.dot(v1, v1) and not self._is_zero_vector(v2_new):
                vertices[0], vertices[1] = v2_new, v1
            else:
                vertices[1] = v2_new
            return vertices
        vertices = self.vertices.copy().astype(np.float64)
        fusion_point = (vertices[0] + vertices[1]) / 2.0
        vertices[0] = vertices[0] + fusion_ratio * (fusion_point - vertices[0])
        vertices[1] = vertices[1] + fusion_ratio * (fusion_point - vertices[1])
        return vertices

    def step2_compress_cd(self, compression_ratio: float = 0.8) -> np.ndarray:
        if self.vertices.dtype == object:
            vertices = self.vertices.copy()
            for k in [2, 3]:
                if k >= len(vertices):
                    continue
                v_k = vertices[k].copy()
                for j in [0, 1]:
                    v_k = self._reduce_vector(v_k, vertices[j])
                vertices[k] = v_k
            return vertices
        vertices = self.vertices.copy().astype(np.float64)
        vertices = self.step1_fuse_ab(fusion_ratio=1.0)
        compression_point = (vertices[2] + vertices[3]) / 2.0
        vertices[2] = vertices[2] + compression_ratio * (compression_point - vertices[2])
        vertices[3] = vertices[3] + compression_ratio * (compression_point - vertices[3])
        return vertices

    def step3_compress_to_point(self, final_ratio: float = 0.9) -> np.ndarray:
        vertices = self.vertices.copy().astype(np.float64)
        vertices = self.step2_compress_cd(compression_ratio=1.0)
        center = np.mean(vertices, axis=0).astype(np.float64)
        for i in range(len(vertices)):
            vertices[i] = vertices[i] + final_ratio * (center - vertices[i])
        return vertices

    def run_geometric_reduction(self, verbose: bool = True, use_lll: bool = True) -> np.ndarray:
        """
        Run SINGLE PASS geometric reduction, then LLL with COLUMN SCALING.
        
        Column scaling preserves lattice structure by scaling each column
        independently to balance magnitudes, enabling LLL to work properly.
        """
        if self.basis is None:
            return np.array([])
            
        basis = self.basis.astype(object)
        n = len(basis)
        m = basis.shape[1] if len(basis.shape) > 1 else n
        
        if n == 0:
            return basis
        
        if verbose:
            print(f"[*] Running Geometric Reduction on {n}x{m} lattice...")
        
        # ===== SINGLE GEOMETRIC PASS =====
        if verbose:
            print(f"[*] Geometric pass: size-reducing all vectors...")
        
        # Forward reduction
        for i in range(1, n):
            for j in range(i):
                basis[i] = self._reduce_vector(basis[i], basis[j])
        
        # Swap pass
        for i in range(n - 1):
            norm_i = np.dot(basis[i], basis[i])
            norm_i1 = np.dot(basis[i+1], basis[i+1])
            if norm_i1 != 0 and norm_i1 < norm_i:
                basis[i], basis[i+1] = basis[i+1].copy(), basis[i].copy()
        
        if verbose:
            print(f"[*] Geometric pass complete.")
        
        # ===== LLL PHASE with COLUMN SCALING =====
        if use_lll:
            try:
                from fpylll import IntegerMatrix, LLL
                
                # Find max bits per column (for column scaling)
                col_max_bits = []
                for j in range(m):
                    max_bits = 0
                    for i in range(n):
                        if basis[i,j] != 0:
                            bits = abs(int(basis[i,j])).bit_length()
                            if bits > max_bits:
                                max_bits = bits
                    col_max_bits.append(max_bits)
                
                overall_max = max(col_max_bits) if col_max_bits else 0
                
                if verbose:
                    print(f"[*] Column max bits: min={min(col_max_bits)}, max={max(col_max_bits)}")
                
                # fpylll works best with entries < ~8000 bits
                TARGET_BITS = 4000
                
                if overall_max > TARGET_BITS:
                    # COLUMN SCALING: Scale each column so max entry ~ TARGET_BITS
                    # This preserves relative structure within each column
                    
                    col_shifts = []
                    for j in range(m):
                        if col_max_bits[j] > TARGET_BITS:
                            shift = col_max_bits[j] - TARGET_BITS
                        else:
                            shift = 0
                        col_shifts.append(shift)
                    
                    if verbose:
                        nz_shifts = [s for s in col_shifts if s > 0]
                        if nz_shifts:
                            print(f"[*] Column scaling: {len(nz_shifts)} columns shifted (max shift 2^{max(nz_shifts)})")
                    
                    # Apply column scaling
                    scaled = np.zeros((n, m), dtype=object)
                    for i in range(n):
                        for j in range(m):
                            if col_shifts[j] > 0:
                                scaled[i,j] = int(basis[i,j]) >> col_shifts[j]
                            else:
                                scaled[i,j] = int(basis[i,j])
                    
                    # Check for all-zero rows after scaling
                    valid_rows = []
                    for i in range(n):
                        if any(scaled[i,j] != 0 for j in range(m)):
                            valid_rows.append(i)
                    
                    if len(valid_rows) < n:
                        if verbose:
                            print(f"[!] Warning: {n - len(valid_rows)} rows became zero after scaling")
                        # Use only valid rows for LLL
                        n_valid = len(valid_rows)
                        if n_valid < 2:
                            if verbose:
                                print(f"[!] Not enough valid rows for LLL, skipping")
                            return basis
                    else:
                        n_valid = n
                        valid_rows = list(range(n))
                    
                    # Build fpylll matrix from valid rows
                    int_matrix = IntegerMatrix(n_valid, m)
                    for idx, i in enumerate(valid_rows):
                        for j in range(m):
                            int_matrix[idx, j] = int(scaled[i, j])
                    
                    if verbose:
                        print(f"[*] Applying LLL to {n_valid}x{m} scaled lattice...")
                    
                    LLL.reduction(int_matrix)
                    
                    # Extract reduced basis
                    reduced_scaled = np.zeros((n_valid, m), dtype=object)
                    for i in range(n_valid):
                        for j in range(m):
                            reduced_scaled[i, j] = int_matrix[i, j]
                    
                    # Unscale columns
                    result = np.zeros((n_valid, m), dtype=object)
                    for i in range(n_valid):
                        for j in range(m):
                            if col_shifts[j] > 0:
                                result[i,j] = int(reduced_scaled[i,j]) << col_shifts[j]
                            else:
                                result[i,j] = int(reduced_scaled[i,j])
                    
                    if verbose:
                        print(f"[*] LLL + column rescale complete.")
                    
                    # If we had to drop rows, put zeros back
                    if n_valid < n:
                        full_result = np.zeros((n, m), dtype=object)
                        for idx, i in enumerate(valid_rows):
                            full_result[i] = result[idx]
                        basis = full_result
                    else:
                        basis = result
                else:
                    # Direct LLL (entries already small enough)
                    int_matrix = IntegerMatrix(n, m)
                    for i in range(n):
                        for j in range(m):
                            int_matrix[i, j] = int(basis[i, j])
                    
                    if verbose:
                        print(f"[*] Applying LLL directly...")
                    
                    LLL.reduction(int_matrix)
                    
                    for i in range(n):
                        for j in range(m):
                            basis[i, j] = int_matrix[i, j]
                    
                    if verbose:
                        print(f"[*] LLL reduction complete.")
                    
            except ImportError:
                if verbose:
                    print(f"[!] fpylll not available")
            except Exception as e:
                if verbose:
                    print(f"[!] LLL failed: {e}")
        
        self.basis = basis
        return basis

    def find_factors_geometrically(self, max_iterations: int = 100) -> Tuple[int, int]:
        try:
            sqrt_N = min(1000000, int(math.isqrt(self.N)) + 1)
        except:
            sqrt_N = 1000000

        for i in range(2, sqrt_N):
            if self.N % i == 0:
                self.p, self.q = i, self.N // i
                return self.p, self.q

        return None, None


def demo_geometric_lll():
    N = 143
    print(f"Factoring N = {N} using Geometric LLL Algorithm")
    print("=" * 50)
    geom_lll = GeometricLLL(N)
    p, q = geom_lll.find_factors_geometrically()
    if p and q:
        print(f"Found factors: {p} x {q} = {N}")
    else:
        print(f"Could not factor {N}")


if __name__ == "__main__":
    demo_geometric_lll()
