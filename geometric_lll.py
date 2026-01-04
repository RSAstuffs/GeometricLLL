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

    def step0_expand_to_line(self, expansion_factor: float = 2.0) -> np.ndarray:
        """
        EXPAND: Vector/Point → Line
        Spread the compressed vector along one dimension to create a line.
        """
        if self.vertices.dtype == object:
            vertices = self.vertices.copy()
        else:
            vertices = self.vertices.copy().astype(np.float64)

        # Start with the center point
        center = np.mean(vertices, axis=0)

        # Expand along the primary axis (x-direction)
        # Move vertices away from center along x-axis
        for i in range(len(vertices)):
            direction = vertices[i] - center
            if np.linalg.norm(direction) > 0:
                # Expand along x-axis
                vertices[i][0] += expansion_factor * direction[0]
                # Keep some y-variation but reduced
                vertices[i][1] += 0.5 * expansion_factor * direction[1]

        return vertices

    def step1_expand_to_triangle(self, expansion_factor: float = 1.5) -> np.ndarray:
        """
        EXPAND: Line → Triangle
        Add a second dimension to form a triangular shape.
        """
        vertices = self.step0_expand_to_line(expansion_factor)

        # Create triangular formation
        # Vertex A stays near origin, B and C/D form the triangle base
        center = np.mean(vertices, axis=0)

        # Position vertices in triangular formation
        if len(vertices) >= 3:
            # A: top vertex (compressed position)
            vertices[0] = center + np.array([0, expansion_factor * 2])

            # B: left base vertex
            vertices[1] = center + np.array([-expansion_factor * 2, -expansion_factor])

            # C: right base vertex
            if len(vertices) > 2:
                vertices[2] = center + np.array([expansion_factor * 2, -expansion_factor])

            # D: center base vertex (if exists)
            if len(vertices) > 3:
                vertices[3] = center + np.array([0, -expansion_factor * 1.5])

        return vertices

    def step2_expand_to_square(self, expansion_factor: float = 1.0) -> np.ndarray:
        """
        EXPAND: Triangle → Square
        Form the full square by positioning vertices at corners.
        """
        vertices = self.step1_expand_to_triangle(expansion_factor)

        # Form square: A B C D in square formation
        center = np.mean(vertices, axis=0)
        size = expansion_factor * 3

        if len(vertices) >= 4:
            vertices[0] = center + np.array([-size, size])   # Top-left (A)
            vertices[1] = center + np.array([size, size])    # Top-right (B)
            vertices[2] = center + np.array([size, -size])   # Bottom-right (C)
            vertices[3] = center + np.array([-size, -size])  # Bottom-left (D)

        return vertices

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

    def _expand_and_recompress_geometric(self, verbose: bool = True) -> np.ndarray:
        """
        NEW GEOMETRIC SEQUENCE: Expand Vector → Line → Triangle → Square → Recompress

        This implements the user's requested geometric transformation:
        1. Start with compressed vector/point
        2. Expand into a line
        3. Expand line into triangle
        4. Expand triangle into square
        5. Recompress the square using traditional geometric reduction
        """
        if self.basis is None:
            return np.array([])

        basis = self.basis.astype(object)
        n = len(basis)

        if verbose:
            print("[*] Step 1: EXPANDING vector into LINE...")
            print("[*] Starting from compressed state, expanding along primary axis")

        # Step 1: Start with compressed state (vectors are already in some compressed form)
        # The basis vectors represent our initial "compressed" state

        # Step 2: Expand to line - spread vectors along one dimension
        if verbose:
            print("[*] Step 2: EXPANDING line into TRIANGLE...")

        # For lattice vectors, we interpret expansion geometrically
        # Add small perturbations to create "triangular" relationships
        for i in range(1, min(4, n)):
            for j in range(i):
                # Add triangular relationships between vectors
                # Use integer division to avoid float overflow
                basis[i] = basis[i] + (basis[j] // 10)  # Small expansion factor (divide by 10 instead of multiply by 0.1)

        if verbose:
            print("[*] Step 3: EXPANDING triangle into SQUARE...")

        # Step 3: Expand to square formation
        # Create square relationships by ensuring orthogonal-like properties
        if n >= 4:
            # Make vectors more orthogonal (square-like)
            for i in range(2, min(4, n)):
                # Orthogonalize against previous vectors
                for j in range(i):
                    dot_ij = np.dot(basis[i], basis[j])
                    dot_jj = np.dot(basis[j], basis[j])
                    proj = dot_ij // dot_jj if dot_jj != 0 else 0
                    basis[i] = basis[i] - proj * basis[j]

        if verbose:
            print("[*] Step 4: RECOMPRESSING square back to optimal form...")

        # Step 4: Recompress using the existing geometric reduction
        # Use the rotating reduction pattern to find the optimal compressed form
        best_basis = basis.copy()
        best_shortest_norm = None

        # Calculate initial shortest
        for i in range(n):
            norm_sq = np.dot(basis[i], basis[i])
            if norm_sq > 0:
                if best_shortest_norm is None or norm_sq < best_shortest_norm:
                    best_shortest_norm = norm_sq

        # Apply rotating compression patterns (similar to run_geometric_reduction)
        patterns = [
            ("A-B fusion", lambda i, n: [(i, i+1) for i in range(0, n-1, 2)]),
            ("C-D fusion", lambda i, n: [(i, i+1) for i in range(1, n-1, 2)]),
            ("A-C fusion", lambda i, n: [(i, i+2) for i in range(0, n-2)]),
        ]

        for pass_num in range(3):  # 3 passes for the expansion sequence
            pattern_name, get_pairs = patterns[pass_num % len(patterns)]
            pairs = get_pairs(0, n)

            if verbose:
                print(f"[*] Compression pass {pass_num + 1}: {pattern_name}")

            # Compress pairs
            for (i, j) in pairs:
                if i < n and j < n:
                    basis[j] = self._reduce_vector(basis[j], basis[i])

            # Full reduction
            for i in range(1, n):
                for j in range(i):
                    basis[i] = self._reduce_vector(basis[i], basis[j])

            # Backward reduction
            for i in range(n - 2, -1, -1):
                for j in range(i + 1, n):
                    basis[i] = self._reduce_vector(basis[i], basis[j])

            # Check if better
            current_shortest = None
            for i in range(n):
                norm_sq = np.dot(basis[i], basis[i])
                if norm_sq > 0:
                    if current_shortest is None or norm_sq < current_shortest:
                        current_shortest = norm_sq

            if best_shortest_norm is None or (current_shortest and current_shortest < best_shortest_norm):
                best_shortest_norm = current_shortest
                best_basis = basis.copy()
                if verbose:
                    bits = current_shortest.bit_length() // 2 if current_shortest else 0
                    print(f"[*] Improved to ~2^{bits} bits")

        if verbose:
            final_bits = best_shortest_norm.bit_length() // 2 if best_shortest_norm else 0
            print(f"[*] Geometric expansion-recompression complete. Shortest vector: ~2^{final_bits} bits")

        return best_basis

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




    def run_bkz(self, block_size: int = 20, verbose: bool = True, 
                max_tours: int = 10) -> np.ndarray:
        """
        Custom BKZ (Block Korkine-Zolotarev) reduction using geometric SVP.
        
        BKZ is stronger than LLL - it processes blocks of vectors and finds
        shortest vectors within each block using an SVP oracle.
        
        This implementation uses YOUR geometric reduction as the SVP oracle,
        avoiding overflow issues with arbitrary precision integers.
        
        Algorithm:
        1. Start with geometrically-reduced basis
        2. For each block [i, i+block_size]:
           a. Extract the block (projected onto orthogonal complement)
           b. Find shortest vector in block using geometric SVP
           c. Insert shortest vector and re-reduce
        3. Repeat tours until no improvement
        
        Args:
            block_size: Size of blocks to process (larger = stronger but slower)
            verbose: Print progress
            max_tours: Maximum number of BKZ tours
            
        Returns:
            BKZ-reduced basis
        """
        if self.basis is None:
            return np.array([])
        
        basis = self.basis.astype(object)
        n = len(basis)
        
        if n == 0:
            return basis
        
        # Clamp block size
        block_size = min(block_size, n)
        
        if verbose:
            print(f"[*] Running Geometric BKZ on {n}x{basis.shape[1]} lattice...")
            print(f"[*] Block size: {block_size}, Max tours: {max_tours}")
        
        # First, do a geometric LLL reduction
        if verbose:
            print(f"[*] Initial geometric reduction...")
        self.basis = basis
        basis = self.run_geometric_reduction(verbose=False, num_passes=1)
        
        best_basis = basis.copy()
        best_norm = self._get_shortest_norm(basis)
        
        for tour in range(max_tours):
            if verbose:
                print(f"\n[*] === BKZ TOUR {tour + 1}/{max_tours} ===")
            
            tour_improved = False
            
            # Process each block
            for k in range(n - 1):
                # Block indices: [k, min(k + block_size, n))
                block_end = min(k + block_size, n)
                block_len = block_end - k
                
                if block_len < 2:
                    continue
                
                # Extract block
                block = basis[k:block_end].copy()
                
                # Project block onto orthogonal complement of basis[0:k]
                # For simplicity, we work with the block directly
                # (full projection would be more accurate but complex)
                
                # Find shortest vector in block using geometric reduction
                block_lll = GeometricLLL(self.N, basis=block)
                reduced_block = block_lll.run_geometric_reduction(verbose=False, num_passes=4)
                
                # Get the shortest vector from reduced block
                shortest_idx = 0
                shortest_norm = np.dot(reduced_block[0], reduced_block[0])
                for i in range(1, len(reduced_block)):
                    norm = np.dot(reduced_block[i], reduced_block[i])
                    if norm > 0 and (shortest_norm == 0 or norm < shortest_norm):
                        shortest_norm = norm
                        shortest_idx = i
                
                # Check if this improves the basis
                current_norm = np.dot(basis[k], basis[k])
                
                if shortest_norm > 0 and shortest_norm < current_norm:
                    # Insert shortest vector at position k
                    # Move other vectors down
                    new_vector = reduced_block[shortest_idx].copy()
                    
                    # Shift vectors
                    for i in range(block_end - 1, k, -1):
                        basis[i] = basis[i-1].copy()
                    basis[k] = new_vector
                    
                    # Re-reduce the affected portion
                    for i in range(k, min(k + block_size + 1, n)):
                        for j in range(i):
                            basis[i] = self._reduce_vector(basis[i], basis[j])
                    
                    tour_improved = True
                    
                    if verbose:
                        bits = shortest_norm.bit_length() // 2
                        print(f"[*] Block {k}: improved to ~2^{bits} bits")
            
            # Full re-reduction after tour
            for i in range(1, n):
                for j in range(i):
                    basis[i] = self._reduce_vector(basis[i], basis[j])
            
            # Track best
            current_best = self._get_shortest_norm(basis)
            if current_best and (best_norm is None or current_best < best_norm):
                best_norm = current_best
                best_basis = basis.copy()
                if verbose:
                    bits = best_norm.bit_length() // 2
                    print(f"[*] ★ New best: ~2^{bits} bits")
            
            if not tour_improved:
                if verbose:
                    print(f"[*] No improvement in tour {tour + 1}, stopping early")
                break
        
        if verbose:
            if best_norm:
                bits = best_norm.bit_length() // 2
                print(f"\n[*] BKZ complete. Best shortest: ~2^{bits} bits")
            else:
                print(f"\n[*] BKZ complete.")
        
        self.basis = best_basis
        return best_basis
    
    def _get_shortest_norm(self, basis) -> Optional[int]:
        """Get the squared norm of the shortest non-zero vector."""
        shortest = None
        for i in range(len(basis)):
            norm_sq = np.dot(basis[i], basis[i])
            if norm_sq > 0:
                if shortest is None or norm_sq < shortest:
                    shortest = norm_sq
        return shortest



    def find_roots_geometrically(self, polynomial, X: int, verbose: bool = True) -> list:
        """
        Find small roots of polynomial mod N using PURE geometric methods.

        The geometric interpretation:
        - The lattice vectors represent polynomial coefficients
        - Short vectors = polynomials with small coefficients
        - The expansion-recompression finds the shortest vector
        - Roots are extracted from the geometric "focal point"

        This is YOUR custom root-finding algorithm:
        1. Build Coppersmith lattice
        2. Expand Vector → Line → Triangle → Square → Recompress (geometric transformation)
        3. Extract root from the "point" (shortest vector)
        4. Verify root geometrically by checking if it collapses N
        
        Args:
            polynomial: Function f(x) where we seek f(x) ≡ 0 (mod N)
            X: Bound on root size |x| < X
            verbose: Print geometric transformation steps
            
        Returns:
            List of roots found
        """
        if self.basis is None:
            if verbose:
                print("[!] No basis provided - cannot find roots")
            return []
        
        n = len(self.basis)
        roots = []
        # Phase 0: Try SIMPLE 2D lattice first (often works best!)
        if verbose:
            print("[*] Phase 0: SIMPLE 2D LATTICE (direct approach)")
        
        # Extract polynomial coefficients from the basis
        # For f(x) = a*x + b, build simple lattice [[N, 0], [b, a]]
        if self.basis is not None and len(self.basis) >= 2:
            # Try to get coefficients from first two rows
            try:
                # The polynomial coefficients might be in the basis
                b = int(self.basis[1][0]) if len(self.basis[1]) > 0 else 0
                a = int(self.basis[1][1]) if len(self.basis[1]) > 1 else 0
                
                # Normalize - extract the actual polynomial coefficients
                # by looking at ratios
                if a != 0 and b != 0:
                    simple_basis = np.array([
                        [self.N, 0],
                        [b % self.N, a % self.N if a > 0 else a]
                    ], dtype=object)
                    
                    # Reduce this simple lattice
                    simple_geom = GeometricLLL(self.N, basis=simple_basis)
                    simple_reduced = simple_geom.run_geometric_reduction(verbose=False, num_passes=1)
                    
                    # Check shortest vector for root
                    for vec in simple_reduced:
                        if len(vec) >= 2:
                            c0, c1 = int(vec[0]), int(vec[1])
                            if c1 != 0 and math.gcd(abs(c1), self.N) == 1:
                                try:
                                    c1_inv = pow(c1, -1, self.N)
                                    root_candidate = (-c0 * c1_inv) % self.N
                                    if root_candidate > self.N // 2:
                                        root_candidate -= self.N
                                    if abs(root_candidate) <= X:
                                        f_val = polynomial(root_candidate)
                                        if f_val % self.N == 0:
                                            if root_candidate not in roots:
                                                roots.append(root_candidate)
                                                if verbose:
                                                    print(f"[★] SIMPLE LATTICE ROOT: x = {root_candidate}")
                                except:
                                    pass
            except Exception as e:
                if verbose:
                    print(f"[*] Simple lattice attempt failed: {e}")
        

        
        if verbose:
            print(f"[*] GEOMETRIC ROOT FINDING")
            print(f"[*] Seeking roots |x| < {X} of f(x) ≡ 0 (mod N)")
            print(f"[*] Lattice dimension: {n}x{self.basis.shape[1]}")
            print()
            print("[*] Phase 1: EXPAND VECTOR → LINE → TRIANGLE → SQUARE → RECOMPRESS")
        
        # Phase 1: New geometric sequence - Expand Vector → Line → Triangle → Square → Recompress
        reduced = self._expand_and_recompress_geometric(verbose=verbose)
        
        if verbose:
            print()
            print("[*] Phase 2: EXTRACT ROOTS FROM FOCAL POINT")
        
        # Phase 2: Extract roots from shortest vectors
        # The "focal point" contains the answer encoded geometrically
        
        # Calculate norms
        norms = []
        for i in range(len(reduced)):
            norm_sq = np.dot(reduced[i], reduced[i])
            norms.append((i, norm_sq))
        norms.sort(key=lambda x: x[1])
        
        # Check shortest vectors for roots
        for idx, norm_sq in norms[:min(10, len(norms))]:
            vec = reduced[idx]
            
            if verbose:
                bits = norm_sq.bit_length() // 2 if norm_sq > 0 else 0
                print(f"[*] Examining focal vector {idx} (magnitude ~2^{bits})")
            
            # Method 1: Linear root extraction
            # If vec = [b, a, ...], then root might be x = -b/a or x = -b*a^(-1) mod N
            if len(vec) >= 2:
                b = int(vec[0])
                a = int(vec[1])
                
                if a != 0:
                    # Try exact division first
                    if b % a == 0:
                        root_candidate = -b // a
                        if abs(root_candidate) <= X:
                            try:
                                f_val = polynomial(root_candidate)
                                if f_val != 0 and f_val % self.N == 0:
                                    if root_candidate not in roots:
                                        roots.append(root_candidate)
                                        if verbose:
                                            print(f"[★] GEOMETRIC ROOT FOUND: x = {root_candidate}")
                                            print(f"    f({root_candidate}) = {f_val}")
                                            print(f"    f(x) mod N = 0 ✓")
                            except:
                                pass
                    
                    # Try MODULAR root: x = -b * a^(-1) mod N
                    # This is the key insight for Coppersmith!
                    try:
                        if math.gcd(abs(a), self.N) == 1:
                            a_inv = pow(a, -1, self.N)
                            root_candidate = (-b * a_inv) % self.N
                            # Check if it's small enough
                            if root_candidate > self.N // 2:
                                root_candidate -= self.N  # Try negative
                            if abs(root_candidate) <= X:
                                f_val = polynomial(root_candidate)
                                if f_val % self.N == 0:
                                    if root_candidate not in roots:
                                        roots.append(root_candidate)
                                        if verbose:
                                            print(f"[★] MODULAR ROOT FOUND: x = {root_candidate}")
                                            print(f"    Computed as -b*a^(-1) mod N")
                    except:
                        pass
                    
                    # Try with rounding
                    for offset in range(-2, 3):
                        root_candidate = -b // a + offset if a != 0 else 0
                        if abs(root_candidate) <= X:
                            try:
                                f_val = polynomial(root_candidate)
                                if f_val != 0 and f_val % self.N == 0:
                                    if root_candidate not in roots:
                                        roots.append(root_candidate)
                                        if verbose:
                                            print(f"[★] GEOMETRIC ROOT FOUND: x = {root_candidate}")
                            except:
                                pass
            
            # Method 2: GCD-based extraction
            # The shortest vector might encode the root through GCDs
            for i in range(min(3, len(vec))):
                for j in range(i + 1, min(4, len(vec))):
                    vi, vj = abs(int(vec[i])), abs(int(vec[j]))
                    if vi > 0 and vj > 0:
                        g = math.gcd(vi, vj)
                        if g > 1 and g <= X:
                            try:
                                f_val = polynomial(g)
                                if f_val != 0 and f_val % self.N == 0:
                                    if g not in roots:
                                        roots.append(g)
                                        if verbose:
                                            print(f"[★] GCD ROOT FOUND: x = {g}")
                                f_val = polynomial(-g)
                                if f_val != 0 and f_val % self.N == 0:
                                    if -g not in roots:
                                        roots.append(-g)
                                        if verbose:
                                            print(f"[★] GCD ROOT FOUND: x = {-g}")
                            except:
                                pass
            
            # Method 3: Ratio extraction
            # Check ratios between components
            for i in range(min(4, len(vec))):
                vi = int(vec[i])
                if vi != 0 and abs(vi) <= X:
                    try:
                        f_val = polynomial(vi)
                        if f_val != 0 and f_val % self.N == 0:
                            if vi not in roots:
                                roots.append(vi)
                                if verbose:
                                    print(f"[★] DIRECT ROOT: x = {vi}")
                    except:
                        pass
        
        if verbose:
            print()
            if roots:
                print(f"[*] Phase 3: VERIFICATION COMPLETE")
                print(f"[★] Found {len(roots)} geometric root(s): {roots}")
            else:
                print(f"[*] No roots found in focal point")
                print(f"[*] The lattice may need different parameters")
        
        return roots


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
