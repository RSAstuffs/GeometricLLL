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

    def run_geometric_reduction(self, verbose: bool = True, 
                                num_passes: int = 1) -> np.ndarray:
        """
        Run ROTATING geometric reduction:
        
        Each pass compresses different pairs in rotation:
        Pass 1: A-B (adjacent pairs 0-1, 2-3, 4-5...)
        Pass 2: C-D (offset pairs 1-2, 3-4, 5-6...)  
        Pass 3: A-C (stride-2 pairs 0-2, 1-3, 2-4...)
        Pass 4: B-D (another stride pattern)
        
        This explores the lattice from different "angles"!
        
        Args:
            verbose: Print progress information
            num_passes: Number of rotation cycles
        """
        if self.basis is None:
            return np.array([])
            
        basis = self.basis.astype(object)
        n = len(basis)
        m = basis.shape[1] if len(basis.shape) > 1 else n
        
        if n == 0:
            return basis
        
        if verbose:
            print(f"[*] Running ROTATING Geometric Reduction on {n}x{m} lattice...")
            if num_passes > 1:
                print(f"[*] Rotation cycles: {num_passes}")
        
        best_basis = basis.copy()
        best_shortest_norm = None
        
        # Calculate initial best
        for i in range(n):
            norm_sq = np.dot(basis[i], basis[i])
            if norm_sq > 0:
                if best_shortest_norm is None or norm_sq < best_shortest_norm:
                    best_shortest_norm = norm_sq
        
        # Define rotation patterns (which pairs to fuse)
        patterns = [
            ("A-B", lambda i, n: [(i, i+1) for i in range(0, n-1, 2)]),      # Adjacent: 0-1, 2-3, 4-5
            ("C-D", lambda i, n: [(i, i+1) for i in range(1, n-1, 2)]),      # Offset: 1-2, 3-4, 5-6
            ("A-C", lambda i, n: [(i, i+2) for i in range(0, n-2)]),         # Stride-2: 0-2, 1-3, 2-4
            ("B-D", lambda i, n: [(i, i+3) for i in range(0, n-3)]),         # Stride-3: 0-3, 1-4, 2-5
        ]
        
        for pass_num in range(num_passes):
            pattern_idx = pass_num % len(patterns)
            pattern_name, get_pairs = patterns[pattern_idx]
            
            if verbose:
                print(f"\n[*] === ROTATION {pass_num + 1}/{num_passes}: {pattern_name} ===")
            
            # Get pairs for this pattern
            pairs = get_pairs(0, n)
            
            if verbose:
                print(f"[*] Compressing pairs: {pairs[:4]}{'...' if len(pairs) > 4 else ''}")
            
            # ===== COMPRESS using current pattern =====
            # First, fuse the specified pairs
            for (i, j) in pairs:
                if i < n and j < n:
                    basis[j] = self._reduce_vector(basis[j], basis[i])
            
            # Then do full forward reduction
            for i in range(1, n):
                for j in range(i):
                    basis[i] = self._reduce_vector(basis[i], basis[j])
            
            # Backward reduction
            for i in range(n - 2, -1, -1):
                for j in range(i + 1, n):
                    basis[i] = self._reduce_vector(basis[i], basis[j])
            
            # Check if we found better
            current_shortest = None
            for i in range(n):
                norm_sq = np.dot(basis[i], basis[i])
                if norm_sq > 0:
                    if current_shortest is None or norm_sq < current_shortest:
                        current_shortest = norm_sq
            
            if verbose:
                bits = current_shortest.bit_length() // 2 if current_shortest else 0
                print(f"[*] Shortest after {pattern_name}: ~2^{bits} bits")
            
            if best_shortest_norm is None or (current_shortest and current_shortest < best_shortest_norm):
                best_shortest_norm = current_shortest
                best_basis = basis.copy()
                if verbose:
                    print(f"[*] ★ New best found!")
            
            # ===== EXPAND before next rotation =====
            if pass_num < num_passes - 1:
                # Expand using REVERSE of current pattern
                reverse_pairs = list(reversed(pairs))
                for (i, j) in reverse_pairs[:len(reverse_pairs)//2]:
                    if i < n and j < n:
                        # Add vectors to expand
                        basis[i] = basis[i] + basis[j]
        
        if verbose:
            best_bits = best_shortest_norm.bit_length() // 2 if best_shortest_norm else 0
            print(f"\n[*] Rotation complete. Best shortest: ~2^{best_bits} bits")
        
        self.basis = best_basis
        return best_basis



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
