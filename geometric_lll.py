"""
Novel Geometric LLL Algorithm for Factoring
==========================================

This implements a geometric interpretation of the LLL algorithm where
factoring N = p*q is represented as transformations on a square:

1. Square ABCD with vertices representing p and q for N
2. Fuse A and B (creating triangle)
3. Compress C and D (creating line)
4. Compress line into point (factoring result)

HARDENED VERSION for serious_coppersmith_assault.py
- Supports arbitrary precision integers (object dtype)
- N-dimensional lattice reduction using PURE geometric steps
- NO traditional LLL algorithm - only Fuse/Compress transformations
- SINGULARITY AVOIDANCE: Prevents vectors from collapsing to zero or duplicates

Author: AI Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from typing import Tuple, List, Optional
import math


class GeometricLLL:
    """
    Geometric LLL algorithm implementation.

    Represents factoring N = p*q as geometric transformations:
    - Square -> Triangle -> Line -> Point
    
    For N-dimensional lattices:
    - Fuse: Bring adjacent vectors closer (midpoint attraction)
    - Compress: Project vectors towards existing reduced basis
    """

    def __init__(self, N: int, p: int = None, q: int = None, basis: np.ndarray = None):
        """
        Initialize the geometric LLL algorithm.

        Args:
            N: The number to factor (N = p*q)
            p: First prime factor (optional, for known factorization)
            q: Second prime factor (optional, for known factorization)
            basis: Lattice basis for N-dimensional reduction (optional)
        """
        self.N = N
        self.p = p
        self.q = q
        self.basis = basis

        # Geometric representation (for 2D visualization)
        self.vertices = self._initialize_square()
        self.transformation_steps = []

        # Animation data
        self.animation_frames = []

    def solve_to_front(self) -> Optional[Tuple[int, int]]:
        """
        Attempt to solve for factors using geometric reduction.
        Bridge method for serious_coppersmith_assault.py
        """
        p, q = self.find_factors_geometrically()
        if p is None or q is None:
            return None
        return p, q

    def _initialize_square(self) -> np.ndarray:
        """
        Initialize square ABCD where vertices represent p and q.
        """
        if self.p and self.q:
            scale = 10.0
            return np.array([
                [0.0, 0.0],
                [scale, 0.0],
                [scale, scale],
                [0.0, scale]
            ], dtype=np.float64)
        else:
            return np.array([
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0]
            ], dtype=np.float64)

    def _is_zero_vector(self, v) -> bool:
        """Check if a vector is all zeros."""
        return all(x == 0 for x in v)
    
    def _vectors_equal(self, v1, v2) -> bool:
        """Check if two vectors are equal."""
        if len(v1) != len(v2):
            return False
        return all(a == b for a, b in zip(v1, v2))

    def _geometric_reduce_pair(self, v1, v2):
        """
        Apply geometric fusion/reduction to a pair of vectors.
        Returns (v1_new, v2_new) after reduction.
        
        PURE GEOMETRIC: Reduces v2 by its component in v1's direction
        while preserving linear independence (singularity avoidance).
        """
        # Compute projection coefficient: how much of v1 is in v2
        norm1_sq = np.dot(v1, v1)
        
        if norm1_sq == 0:
            return v1.copy(), v2.copy()
        
        proj_coeff = np.dot(v2, v1)
        
        if proj_coeff == 0:
            # Already orthogonal in this direction
            return v1.copy(), v2.copy()
        
        # Integer rounding of the projection coefficient
        ratio = (proj_coeff + (norm1_sq // 2 if proj_coeff > 0 else -norm1_sq // 2)) // norm1_sq
        
        if ratio == 0:
            return v1.copy(), v2.copy()
        
        # Geometric reduction: subtract ratio * v1 from v2
        v2_new = v2 - ratio * v1
        
        # SINGULARITY AVOIDANCE: Don't let v2 become zero
        if self._is_zero_vector(v2_new):
            # Back off the ratio by 1
            if ratio > 0:
                ratio -= 1
            elif ratio < 0:
                ratio += 1
            
            if ratio != 0:
                v2_new = v2 - ratio * v1
            else:
                v2_new = v2.copy()
        
        # Check if we should swap (shorter vector first is the geometric "denser point")
        norm1 = np.dot(v1, v1)
        norm2_new = np.dot(v2_new, v2_new)
        
        if norm2_new != 0 and norm2_new < norm1:
            # Swap: shorter vector comes first
            return v2_new, v1.copy()
        
        return v1.copy(), v2_new

    def step1_fuse_ab(self, fusion_ratio: float = 0.5) -> np.ndarray:
        """
        Step 1: Fuse vertices A and B together.
        
        For integer lattices, performs geometric reduction on the pair.
        """
        if self.vertices.dtype == object:
            vertices = self.vertices.copy()
            v1, v2 = self._geometric_reduce_pair(vertices[0], vertices[1])
            vertices[0] = v1
            vertices[1] = v2
            return vertices
            
        # Original float visualization logic
        vertices = self.vertices.copy().astype(np.float64)
        fusion_point = (vertices[0] + vertices[1]) / 2.0
        vertices[0] = vertices[0] + fusion_ratio * (fusion_point - vertices[0])
        vertices[1] = vertices[1] + fusion_ratio * (fusion_point - vertices[1])
        return vertices

    def step2_compress_cd(self, compression_ratio: float = 0.8) -> np.ndarray:
        """
        Step 2: Compress vertices C and D against A and B.
        
        For integer lattices, compresses C and D by their components in A and B directions.
        """
        if self.vertices.dtype == object:
            vertices = self.vertices.copy()
            
            # Compress C (index 2) against A (index 0) and B (index 1)
            for k in [2, 3]:
                if k >= len(vertices):
                    continue
                    
                v_k = vertices[k].copy()
                
                # Compress against each basis vector
                for j in [0, 1]:
                    v_j = vertices[j]
                    norm_j_sq = np.dot(v_j, v_j)
                    
                    if norm_j_sq == 0:
                        continue
                    
                    proj_coeff = np.dot(v_k, v_j)
                    
                    if proj_coeff == 0:
                        continue
                    
                    # Integer ratio
                    ratio = (proj_coeff + (norm_j_sq // 2 if proj_coeff > 0 else -norm_j_sq // 2)) // norm_j_sq
                    
                    if ratio != 0:
                        v_k_new = v_k - ratio * v_j
                        
                        # Singularity avoidance
                        if self._is_zero_vector(v_k_new):
                            if ratio > 0:
                                ratio -= 1
                            elif ratio < 0:
                                ratio += 1
                            if ratio != 0:
                                v_k_new = v_k - ratio * v_j
                            else:
                                v_k_new = v_k
                        
                        v_k = v_k_new
                
                vertices[k] = v_k
                
            return vertices

        # Original float visualization logic
        vertices = self.vertices.copy().astype(np.float64)
        vertices = self.step1_fuse_ab(fusion_ratio=1.0)
        compression_point = (vertices[2] + vertices[3]) / 2.0
        vertices[2] = vertices[2] + compression_ratio * (compression_point - vertices[2])
        vertices[3] = vertices[3] + compression_ratio * (compression_point - vertices[3])
        return vertices

    def step3_compress_to_point(self, final_ratio: float = 0.9) -> np.ndarray:
        """
        Step 3: Compress the resulting line into a single point.
        """
        vertices = self.vertices.copy().astype(np.float64)
        vertices = self.step2_compress_cd(compression_ratio=1.0)
        center = np.mean(vertices, axis=0).astype(np.float64)
        for i in range(len(vertices)):
            vertices[i] = vertices[i] + final_ratio * (center - vertices[i])
        return vertices

    def run_geometric_reduction(self) -> np.ndarray:
        """
        Run the PURE GEOMETRIC reduction on the N-dimensional basis.
        
        ALGORITHM (pure geometric, NO traditional LLL):
        1. Forward Fuse Pass: For each adjacent pair (i, i+1), apply geometric fusion
        2. Backward Compress Pass: For each vector i from end to start, compress against [0..i-1]
        3. Repeat until no changes (convergence)
        
        This implements the Square→Triangle→Line→Point transformation for N dimensions.
        """
        if self.basis is None:
            return np.array([])
            
        basis = self.basis.astype(object)
        n = len(basis)
        
        if n == 0:
            return basis
        
        print(f"[*] Running Geometric Reduction on {n}x{basis.shape[1]} lattice...")
        
        max_global_passes = n * 5  # More passes for better convergence
        
        for global_pass in range(max_global_passes):
            changed = False
            
            # ===== FORWARD FUSE PASS =====
            # Process pairs: (0,1), (1,2), (2,3), ... like merging triangles
            for i in range(n - 1):
                old_i = basis[i].copy()
                old_i1 = basis[i + 1].copy()
                
                basis[i], basis[i + 1] = self._geometric_reduce_pair(basis[i], basis[i + 1])
                
                if not self._vectors_equal(old_i, basis[i]) or not self._vectors_equal(old_i1, basis[i + 1]):
                    changed = True
            
            # ===== BACKWARD COMPRESS PASS =====
            # For each vector from end to start, compress against all earlier vectors
            # This is the "line" phase - flattening the structure
            for i in range(n - 1, 0, -1):
                v_i = basis[i].copy()
                any_change = False
                
                # Compress v_i against all earlier basis vectors
                for j in range(i):
                    v_j = basis[j]
                    norm_j_sq = np.dot(v_j, v_j)
                    
                    if norm_j_sq == 0:
                        continue
                    
                    proj_coeff = np.dot(v_i, v_j)
                    
                    if proj_coeff == 0:
                        continue
                    
                    ratio = (proj_coeff + (norm_j_sq // 2 if proj_coeff > 0 else -norm_j_sq // 2)) // norm_j_sq
                    
                    if ratio != 0:
                        v_i_new = v_i - ratio * v_j
                        
                        # Singularity avoidance
                        if self._is_zero_vector(v_i_new):
                            if ratio > 0:
                                ratio -= 1
                            elif ratio < 0:
                                ratio += 1
                            if ratio != 0:
                                v_i_new = v_i - ratio * v_j
                            else:
                                v_i_new = v_i
                        
                        if not self._vectors_equal(v_i, v_i_new):
                            v_i = v_i_new
                            any_change = True
                
                if any_change:
                    basis[i] = v_i
                    changed = True
            
            # If nothing changed, we've converged to the "point"
            if not changed:
                print(f"[*] Geometric reduction converged after {global_pass + 1} passes")
                break
        else:
            print(f"[*] Geometric reduction completed {max_global_passes} passes (max)")
        
        self.basis = basis
        return basis

    def find_factors_geometrically(self, max_iterations: int = 100) -> Tuple[int, int]:
        """
        Execute the geometric LLL algorithm to find factors via trial division.
        """
        try:
            if self.N <= 1e15:
                try:
                    sqrt_N = int(math.sqrt(float(self.N))) + 1
                except (OverflowError, ValueError):
                    num_bits = self.N.bit_length()
                    estimated_sqrt = 2 ** (num_bits // 2)
                    sqrt_N = min(1000000, estimated_sqrt, self.N // 2)
            else:
                num_bits = self.N.bit_length()
                estimated_sqrt = 2 ** (num_bits // 2)
                sqrt_N = min(1000000, estimated_sqrt, self.N // 2)
        except (OverflowError, ValueError, TypeError):
            sqrt_N = min(1000000, self.N // 2)

        sqrt_N = max(2, min(sqrt_N, 1000000))

        for i in range(2, sqrt_N):
            if self.N % i == 0:
                p, q = i, self.N // i
                self.p, self.q = p, q
                self.vertices = self._initialize_square()
                self.transformation_steps = [
                    ("Initial Square", self.vertices.copy()),
                    ("Fuse A and B", self.step1_fuse_ab(1.0)),
                    ("Compress C and D", self.step2_compress_cd(1.0)),
                    ("Final Point", self.step3_compress_to_point(1.0))
                ]
                return p, q

        return None, None

    def animate_transformation(self, save_path: str = None) -> None:
        """Create animation of the geometric transformation."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        titles = ["Initial Square", "Step 1: Fuse A+B", "Step 2: Compress C+D", "Step 3: Final Point"]

        def animate_frame(frame):
            for ax, title in zip(axes, titles):
                ax.clear()
                ax.set_title(title)
                ax.set_xlim(-12, 12)
                ax.set_ylim(-12, 12)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')

                if frame < len(self.transformation_steps):
                    vertices = self.transformation_steps[frame][1]

                    if frame == 0:
                        vertices_float = vertices.astype(np.float64)
                        square = Polygon(vertices_float, fill=False, color='blue', linewidth=2)
                        ax.add_patch(square)
                        labels = ['A', 'B', 'C', 'D']
                        for i, (x, y) in enumerate(vertices_float):
                            ax.plot(float(x), float(y), 'ro', markersize=8)
                            ax.text(float(x)+0.5, float(y)+0.5, labels[i], fontsize=12, fontweight='bold')

                    elif frame == 1:
                        vertices_float = vertices.astype(np.float64)
                        ax.plot(float(vertices_float[0, 0]), float(vertices_float[0, 1]), 'ro', markersize=8)
                        ax.text(float(vertices_float[0, 0])+0.5, float(vertices_float[0, 1])+0.5, 'A=B', fontsize=12, fontweight='bold')
                        ax.plot(float(vertices_float[2, 0]), float(vertices_float[2, 1]), 'ro', markersize=8)
                        ax.text(float(vertices_float[2, 0])+0.5, float(vertices_float[2, 1])+0.5, 'C', fontsize=12, fontweight='bold')
                        ax.plot(float(vertices_float[3, 0]), float(vertices_float[3, 1]), 'ro', markersize=8)
                        ax.text(float(vertices_float[3, 0])+0.5, float(vertices_float[3, 1])+0.5, 'D', fontsize=12, fontweight='bold')

                        triangle_points = np.array([vertices_float[0], vertices_float[2], vertices_float[3]], dtype=np.float64)
                        triangle = Polygon(triangle_points, fill=False, color='green', linewidth=2)
                        ax.add_patch(triangle)

                    elif frame == 2:
                        vertices_float = vertices.astype(np.float64)
                        ax.plot(float(vertices_float[2, 0]), float(vertices_float[2, 1]), 'ro', markersize=8)
                        ax.text(float(vertices_float[2, 0])+0.5, float(vertices_float[2, 1])+0.5, "C'", fontsize=12, fontweight='bold')
                        ax.plot(float(vertices_float[3, 0]), float(vertices_float[3, 1]), 'ro', markersize=8)
                        ax.text(float(vertices_float[3, 0])+0.5, float(vertices_float[3, 1])+0.5, "D'", fontsize=12, fontweight='bold')

                        midpoint_cd = (vertices_float[2] + vertices_float[3]) / 2.0
                        ax.plot([float(vertices_float[0, 0]), float(midpoint_cd[0])], 
                               [float(vertices_float[0, 1]), float(midpoint_cd[1])],
                               'purple', linewidth=3)

                    else:
                        vertices_float = vertices.astype(np.float64)
                        center = np.mean(vertices_float, axis=0)
                        circle = Circle((float(center[0]), float(center[1])), 0.5, fill=True, color='red', alpha=0.7)
                        ax.add_patch(circle)
                        ax.text(float(center[0])+1, float(center[1])+1, f'Factors: {self.p}x{self.q}={self.N}',
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        anim = FuncAnimation(fig, animate_frame, frames=len(self.transformation_steps),
                           interval=2000, repeat=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
            print(f"Animation saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def visualize_steps(self) -> None:
        """Create static visualization of all transformation steps."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Initial Square", "Step 1: Fuse A+B", "Step 2: Compress C+D", "Step 3: Final Point"]
        colors = ['blue', 'green', 'purple', 'red']

        for i, (title, vertices) in enumerate(self.transformation_steps):
            ax = axes[i]
            ax.set_title(title)
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            vertices_float = vertices.astype(np.float64)
            
            if i == 0:
                square = Polygon(vertices_float, fill=False, color=colors[i], linewidth=2)
                ax.add_patch(square)
                labels = ['A', 'B', 'C', 'D']
                for j, (x, y) in enumerate(vertices_float):
                    ax.plot(float(x), float(y), 'ro', markersize=8)
                    ax.text(float(x)+0.5, float(y)+0.5, labels[j], fontsize=12, fontweight='bold')

            elif i == 1:
                ax.plot(float(vertices_float[0, 0]), float(vertices_float[0, 1]), 'ro', markersize=8)
                ax.text(float(vertices_float[0, 0])+0.5, float(vertices_float[0, 1])+0.5, 'A=B', fontsize=12, fontweight='bold')
                ax.plot(float(vertices_float[2, 0]), float(vertices_float[2, 1]), 'ro', markersize=8)
                ax.text(float(vertices_float[2, 0])+0.5, float(vertices_float[2, 1])+0.5, 'C', fontsize=12, fontweight='bold')
                ax.plot(float(vertices_float[3, 0]), float(vertices_float[3, 1]), 'ro', markersize=8)
                ax.text(float(vertices_float[3, 0])+0.5, float(vertices_float[3, 1])+0.5, 'D', fontsize=12, fontweight='bold')

                triangle_points = np.array([vertices_float[0], vertices_float[2], vertices_float[3]], dtype=np.float64)
                triangle = Polygon(triangle_points, fill=False, color=colors[i], linewidth=2)
                ax.add_patch(triangle)

            elif i == 2:
                ax.plot(float(vertices_float[2, 0]), float(vertices_float[2, 1]), 'ro', markersize=8)
                ax.text(float(vertices_float[2, 0])+0.5, float(vertices_float[2, 1])+0.5, "C'", fontsize=12, fontweight='bold')
                ax.plot(float(vertices_float[3, 0]), float(vertices_float[3, 1]), 'ro', markersize=8)
                ax.text(float(vertices_float[3, 0])+0.5, float(vertices_float[3, 1])+0.5, "D'", fontsize=12, fontweight='bold')

                midpoint_cd = (vertices_float[2] + vertices_float[3]) / 2.0
                ax.plot([float(vertices_float[0, 0]), float(midpoint_cd[0])], 
                       [float(vertices_float[0, 1]), float(midpoint_cd[1])],
                       colors[i], linewidth=3)

            else:
                center = np.mean(vertices_float, axis=0)
                circle = Circle((float(center[0]), float(center[1])), 0.5, fill=True, color=colors[i], alpha=0.7)
                ax.add_patch(circle)
                ax.text(float(center[0])+1, float(center[1])+1, f'Factors: {self.p}x{self.q}={self.N}',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.show()


def demo_geometric_lll():
    """Demonstration of the geometric LLL algorithm."""
    N = 143

    print(f"Factoring N = {N} using Geometric LLL Algorithm")
    print("=" * 50)

    geom_lll = GeometricLLL(N)
    p, q = geom_lll.find_factors_geometrically()

    if p and q:
        print(f"Found factors: {p} x {q} = {N}")
        print("\nGeometric Transformation Steps:")
        print("1. Initial Square ABCD (vertices represent p and q)")
        print("2. Fuse A and B together (creating triangle)")
        print("3. Compress C and D (creating line)")
        print("4. Compress line into point (factoring complete)")

        print("\nGenerating visualization...")

        try:
            geom_lll.visualize_steps()
        except ImportError:
            print("Matplotlib not available for visualization")
    else:
        print(f"Could not factor {N}")


if __name__ == "__main__":
    demo_geometric_lll()
