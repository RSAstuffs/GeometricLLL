#!/usr/bin/env python3
"""
Coppersmith Attack GUI - Minimal Interface
==========================================
A simple tkinter GUI for running Coppersmith attacks with customizable parameters.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import sys
import io
import time

# Increase integer string conversion limit
sys.set_int_max_str_digits(100000)


class CoppersmithGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coppersmith Attack Tool")
        self.root.geometry("900x700")
        
        self.running = False
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # === INPUT SECTION ===
        ttk.Label(main_frame, text="RSA Modulus N:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=row, column=0, sticky="nw", pady=5)
        
        self.n_text = scrolledtext.ScrolledText(main_frame, height=4, width=80)
        self.n_text.grid(row=row, column=1, sticky="ew", pady=5)
        row += 1
        
        # === PARAMETERS SECTION ===
        params_frame = ttk.LabelFrame(main_frame, text="Attack Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        # Lattice parameter m
        ttk.Label(params_frame, text="Lattice m:").grid(row=0, column=0, sticky="w", padx=5)
        self.m_var = tk.StringVar(value="8")
        self.m_spin = ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.m_var, width=10)
        self.m_spin.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(params_frame, text="(larger = more rows, slower)").grid(row=0, column=2, sticky="w")
        
        # Max modulus bits for relationship generation
        ttk.Label(params_frame, text="Max Mod Bits:").grid(row=1, column=0, sticky="w", padx=5)
        self.max_mod_bits_var = tk.StringVar(value="50")
        self.max_mod_spin = ttk.Spinbox(params_frame, from_=10, to=100, textvariable=self.max_mod_bits_var, width=10)
        self.max_mod_spin.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(params_frame, text="(for relationship generation)").grid(row=1, column=2, sticky="w")
        
        # Max relationships
        ttk.Label(params_frame, text="Max Relations:").grid(row=2, column=0, sticky="w", padx=5)
        self.max_rels_var = tk.StringVar(value="5000")
        self.max_rels_entry = ttk.Entry(params_frame, textvariable=self.max_rels_var, width=12)
        self.max_rels_entry.grid(row=2, column=1, sticky="w", padx=5)
        
        # Search bound multiplier
        ttk.Label(params_frame, text="Bound Mult:").grid(row=0, column=3, sticky="w", padx=5)
        self.bound_mult_var = tk.StringVar(value="2")
        self.bound_mult_spin = ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.bound_mult_var, width=10)
        self.bound_mult_spin.grid(row=0, column=4, sticky="w", padx=5)
        
        # Verbose checkbox
        self.verbose_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Verbose Output", variable=self.verbose_var).grid(
            row=1, column=3, columnspan=2, sticky="w", padx=5)
        
        # Use geometric LLL checkbox
        self.use_geom_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Try Geometric LLL First", variable=self.use_geom_var).grid(
            row=2, column=3, columnspan=2, sticky="w", padx=5)
        
        # === BUTTONS ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        self.run_btn = ttk.Button(btn_frame, text="▶ Run Attack", command=self.run_attack)
        self.run_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop_attack, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        ttk.Button(btn_frame, text="Clear Output", command=self.clear_output).pack(side="left", padx=5)
        
        ttk.Button(btn_frame, text="Load N from File", command=self.load_n_file).pack(side="left", padx=5)
        
        ttk.Button(btn_frame, text="Generate Only", command=self.generate_only).pack(side="left", padx=5)
        
        # === OUTPUT SECTION ===
        ttk.Label(main_frame, text="Output:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=row, column=0, sticky="nw", pady=5)
        row += 1
        
        self.output_text = scrolledtext.ScrolledText(main_frame, height=25, width=100, 
                                                      bg='black', fg='lime', font=('Courier', 10))
        self.output_text.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=5)
        main_frame.rowconfigure(row, weight=1)
        row += 1
        
        # === STATUS BAR ===
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=row, column=0, columnspan=2, sticky="ew")
        
    def log(self, msg):
        """Thread-safe logging to output text widget"""
        self.output_text.insert(tk.END, msg + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        
    def load_n_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                self.n_text.delete(1.0, tk.END)
                self.n_text.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                
    def get_n(self):
        """Parse N from input"""
        n_str = self.n_text.get(1.0, tk.END).strip()
        if not n_str:
            raise ValueError("N is empty")
        # Remove whitespace and try to parse
        n_str = ''.join(n_str.split())
        if n_str.startswith('0x'):
            return int(n_str, 16)
        return int(n_str)
    
    def stop_attack(self):
        self.running = False
        self.status_var.set("Stopping...")
        
    def run_attack(self):
        if self.running:
            return
        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Running attack...")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_attack_thread, daemon=True)
        thread.start()
        
    def _run_attack_thread(self):
        try:
            N = self.get_n()
            m = int(self.m_var.get())
            max_mod_bits = int(self.max_mod_bits_var.get())
            max_rels = int(self.max_rels_var.get())
            bound_mult = int(self.bound_mult_var.get())
            verbose = self.verbose_var.get()
            use_geom = self.use_geom_var.get()
            
            self.log(f"=" * 60)
            self.log(f"COPPERSMITH ATTACK")
            self.log(f"=" * 60)
            self.log(f"N: {N.bit_length()}-bit number")
            self.log(f"Parameters: m={m}, max_mod_bits={max_mod_bits}, max_rels={max_rels}")
            self.log("")
            
            # Import modules
            self.log("[*] Loading modules...")
            
            import os
            import sys
            import importlib.util
            
            # Add path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            
            from coppersmith import CoppersmithMethod
            
            # Load geometric LLL
            _module_path = os.path.join(script_dir, 'geometric_lll.py')
            spec = importlib.util.spec_from_file_location('geometric_lll', _module_path)
            geometric_lll_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(geometric_lll_module)
            GeometricLLL = geometric_lll_module.GeometricLLL
            
            # Try geometric LLL first
            if use_geom and self.running:
                self.log("\n[*] Phase 1: Trying Geometric LLL factoring...")
                try:
                    geom_lll = GeometricLLL(N)
                    result = geom_lll.solve_to_front()
                    if result:
                        p, q = result
                        self.log(f"\n🎊 SUCCESS! Found factors via Geometric LLL!")
                        self.log(f"   p = {p}")
                        self.log(f"   q = {q}")
                        self.log(f"   Verification: p*q == N: {p*q == N}")
                        self.status_var.set("SUCCESS!")
                        return
                    else:
                        self.log("   No factors found via Geometric LLL")
                except Exception as e:
                    self.log(f"   Geometric LLL error: {e}")
            
            if not self.running:
                self.log("\n[!] Attack stopped by user")
                return
                
            # Generate relationships
            self.log("\n[*] Phase 2: Generating modular relationships...")
            relationships = self._generate_relationships(N, max_mod_bits, max_rels)
            
            if not relationships:
                self.log("[!] No relationships generated")
                return
                
            self.log(f"   Generated {len(relationships)} relationships")
            
            if not self.running:
                self.log("\n[!] Attack stopped by user")
                return
            
            # Construct polynomial
            self.log("\n[*] Phase 3: Constructing polynomial from constraints...")
            polynomial = self._construct_polynomial(N, relationships)
            
            if polynomial is None:
                self.log("[!] Failed to construct polynomial")
                return
                
            if not self.running:
                self.log("\n[!] Attack stopped by user")
                return
            
            # Run Coppersmith
            self.log("\n[*] Phase 4: Running Coppersmith attack...")
            import math
            search_bound = math.isqrt(math.isqrt(N)) * bound_mult
            
            self.log(f"   Search bound: ~2^{search_bound.bit_length()} ({search_bound:.2e})")
            self.log(f"   Lattice dimension: {m+1}D")
            
            method = CoppersmithMethod(N, polynomial, degree=1)
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = OutputCapture(self.log)
            
            try:
                roots = method.find_small_roots(X=search_bound, m=m, verbose=verbose)
            finally:
                sys.stdout = old_stdout
            
            self.log(f"\n[*] Found {len(roots)} root(s)")
            
            # Check roots for factors
            if roots:
                self.log("\n[*] Checking roots for factors...")
                for root in roots:
                    root = abs(int(root))
                    if root <= 1:
                        continue
                    # Try polynomial value
                    p_candidate = polynomial(root)
                    if p_candidate > 1 and N % p_candidate == 0:
                        q = N // p_candidate
                        self.log(f"\n🎊 SUCCESS! Found factorization!")
                        self.log(f"   p = {p_candidate}")
                        self.log(f"   q = {q}")
                        self.status_var.set("SUCCESS!")
                        return
            
            self.log("\n❌ No factorization found")
            self.status_var.set("Completed - no factors found")
            
        except Exception as e:
            self.log(f"\n❌ Error: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {e}")
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
            
    def _generate_relationships(self, N, max_mod_bits, max_rels):
        """Generate modular relationships"""
        import random
        
        relationships = []
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # Prime powers
        for p in small_primes:
            power = 1
            while True:
                M = p ** power
                if M.bit_length() > max_mod_bits:
                    break
                rem = N % M
                rem_bits = rem.bit_length() if rem > 0 else 0
                mod_bits = M.bit_length()
                if rem_bits < mod_bits - 5:
                    relationships.append((M, rem, rem_bits, mod_bits))
                power += 1
        
        # Smooth numbers
        smooth = []
        def gen_smooth(current, idx):
            if current.bit_length() > max_mod_bits:
                return
            if current > 1:
                smooth.append(current)
            for i in range(idx, len(small_primes)):
                gen_smooth(current * small_primes[i], i)
        gen_smooth(1, 0)
        
        for M in smooth:
            rem = N % M
            rem_bits = rem.bit_length() if rem > 0 else 0
            mod_bits = M.bit_length()
            if rem_bits < mod_bits - 3:
                relationships.append((M, rem, rem_bits, mod_bits))
        
        # Remove duplicates and sort
        seen = set()
        unique = []
        for r in relationships:
            if r[0] not in seen:
                seen.add(r[0])
                unique.append(r)
        
        unique.sort(key=lambda x: (x[1], x[2]))
        return unique[:max_rels]
    
    def _construct_polynomial(self, N, relationships):
        """Construct polynomial from relationships using CRT"""
        try:
            import sympy
            from sympy.ntheory.modular import crt
        except ImportError:
            self.log("[!] SymPy required for CRT")
            return None
        
        sorted_rels = sorted(relationships, key=lambda r: r[0], reverse=True)
        mod_to_rem = {r[0]: r[1] for r in sorted_rels}
        
        used_primes = {}
        selected_moduli = []
        selected_remainders = []
        combined_mod = 1
        
        for M in sorted(mod_to_rem.keys(), reverse=True):
            rem = mod_to_rem[M]
            try:
                factors = sympy.factorint(M)
            except:
                continue
            
            for p, k in factors.items():
                pk = p ** k
                rem_pk = rem % pk
                
                if p in used_primes:
                    old_k, old_rem = used_primes[p]
                    old_pk = p ** old_k
                    if k <= old_k:
                        if old_rem % pk != rem_pk:
                            continue
                    else:
                        if rem_pk % old_pk != old_rem:
                            continue
                        idx = selected_moduli.index(old_pk)
                        selected_moduli.pop(idx)
                        selected_remainders.pop(idx)
                        combined_mod //= old_pk
                        selected_moduli.append(pk)
                        selected_remainders.append(rem_pk)
                        combined_mod *= pk
                        used_primes[p] = (k, rem_pk)
                else:
                    selected_moduli.append(pk)
                    selected_remainders.append(rem_pk)
                    combined_mod *= pk
                    used_primes[p] = (k, rem_pk)
        
        self.log(f"   Selected {len(selected_moduli)} consistent prime powers")
        self.log(f"   Combined modulus: {combined_mod.bit_length()} bits")
        
        if len(selected_moduli) < 2:
            return None
        
        try:
            R_agg, M_agg = crt(selected_moduli, selected_remainders)
        except Exception as e:
            self.log(f"   CRT error: {e}")
            return None
        
        self.log(f"   Polynomial: f(x) = M_agg * x + R_agg")
        
        def polynomial(x):
            return M_agg * x + R_agg
        
        polynomial.coeffs = [int(R_agg), int(M_agg)]
        polynomial.M_agg = int(M_agg)
        polynomial.R_agg = int(R_agg)
        
        return polynomial
    
    def generate_only(self):
        """Just generate and show relationships without running attack"""
        try:
            N = self.get_n()
            max_mod_bits = int(self.max_mod_bits_var.get())
            max_rels = int(self.max_rels_var.get())
            
            self.log(f"Generating relationships for {N.bit_length()}-bit N...")
            relationships = self._generate_relationships(N, max_mod_bits, max_rels)
            
            self.log(f"\nGenerated {len(relationships)} relationships:")
            self.log("-" * 50)
            for i, (M, rem, rem_bits, mod_bits) in enumerate(relationships[:20]):
                self.log(f"{i+1}. M={M} ({mod_bits}b), rem={rem} ({rem_bits}b)")
            if len(relationships) > 20:
                self.log(f"... and {len(relationships)-20} more")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))


class OutputCapture:
    """Capture stdout and redirect to log function"""
    def __init__(self, log_func):
        self.log_func = log_func
        self.buffer = ""
        
    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.log_func(line)
            self.buffer = lines[-1]
            
    def flush(self):
        if self.buffer.strip():
            self.log_func(self.buffer)
            self.buffer = ""


def main():
    root = tk.Tk()
    app = CoppersmithGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
