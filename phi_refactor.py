#!/usr/bin/env python3
"""
phi_refactor.py — φ‑Coherent Complexity Reduction for C++/CUDA Codebases

Applies the 20 formulas from the Simplicity Codex (Formulas 151–170) to analyze
and reduce complexity in DeepGEMM‑style libraries.

Usage:
    python phi_refactor.py --path /path/to/DeepGEMM --analyze
    python phi_refactor.py --path /path/to/DeepGEMM --apply --safe
"""

import os
import re
import sys
import json
import math
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

# ============================================================================
# Golden Ratio Constants
# ============================================================================
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# ============================================================================
# Data Structures for Code Analysis
# ============================================================================
@dataclass
class FunctionInfo:
    name: str
    lines: int
    cyclomatic: int
    params: List[str]
    template_depth: int
    is_public: bool
    file_path: str
    line_start: int
    line_end: int

@dataclass
class FileInfo:
    path: str
    lines: int
    functions: List[FunctionInfo] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    macros: List[Tuple[str, int]] = field(default_factory=list)
    templates: List[Tuple[str, int]] = field(default_factory=list)

@dataclass
class CodebaseMetrics:
    total_lines: int
    total_files: int
    total_functions: int
    public_functions: int
    internal_functions: int
    macro_count: int
    template_instantiations: int
    avg_cyclomatic: float
    include_graph: Dict[str, Set[str]]
    file_infos: List[FileInfo]

# ============================================================================
# Parsing Utilities (Simplified C++/CUDA Parser)
# ============================================================================
class SimpleCppParser:
    """A lightweight regex-based parser for C++/CUDA code analysis."""
    
    def __init__(self):
        self.func_pattern = re.compile(
            r'^\s*(?:template\s*<[^>]*>\s*)?'  # optional template
            r'(?:__host__\s+__device__|__global__|__device__|static\s+)?'  # CUDA specifiers
            r'(?:inline\s+)?(?:constexpr\s+)?(?:virtual\s+)?'
            r'([\w:]+(?:<[^>]*>)?[\s*&]+)'  # return type
            r'(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*'  # name and params
            r'(?:override|final)?\s*\{',
            re.MULTILINE
        )
        self.include_pattern = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]')
        self.macro_pattern = re.compile(r'^\s*#define\s+(\w+)(?:\([^)]*\))?\s+')
        self.template_pattern = re.compile(r'template\s*<([^>]*)>')
    
    def parse_file(self, filepath: str) -> FileInfo:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        lines = content.split('\n')
        
        file_info = FileInfo(
            path=filepath,
            lines=len(lines),
        )
        
        # Extract includes
        for line in lines:
            m = self.include_pattern.match(line.strip())
            if m:
                file_info.includes.append(m.group(1))
        
        # Extract macros
        for i, line in enumerate(lines):
            m = self.macro_pattern.match(line.strip())
            if m:
                file_info.macros.append((m.group(1), i+1))
        
        # Extract templates (simplified)
        for i, line in enumerate(lines):
            if 'template<' in line:
                file_info.templates.append((line.strip(), i+1))
        
        # Extract functions (very simplified)
        funcs = []
        for match in self.func_pattern.finditer(content):
            ret_type = match.group(1).strip()
            name = match.group(2)
            params = match.group(3)
            
            # Rough line count
            start_pos = match.start()
            end_pos = content.find('}', match.end()) + 1
            if end_pos > 0:
                func_content = content[start_pos:end_pos]
                func_lines = func_content.count('\n')
            else:
                func_lines = 1
            
            # Public? (heuristic: in header or not static)
            is_public = filepath.endswith('.h') or filepath.endswith('.hpp') or filepath.endswith('.cuh')
            is_public = is_public and 'static' not in match.group(0)
            
            # Template depth
            template_depth = match.group(0).count('template<')
            
            # Cyclomatic (simplified: count branches)
            if end_pos > 0:
                cyclomatic = 1 + func_content.count('if') + func_content.count('for') + \
                             func_content.count('while') + func_content.count('switch') + \
                             func_content.count('&&') + func_content.count('||')
            else:
                cyclomatic = 1
            
            func_info = FunctionInfo(
                name=name,
                lines=func_lines,
                cyclomatic=cyclomatic,
                params=[p.strip() for p in params.split(',') if p.strip()],
                template_depth=template_depth,
                is_public=is_public,
                file_path=filepath,
                line_start=content[:start_pos].count('\n') + 1,
                line_end=content[:end_pos].count('\n') + 1
            )
            funcs.append(func_info)
        
        file_info.functions = funcs
        return file_info

# ============================================================================
# Formula Implementations (151–170)
# ============================================================================
class PhiRefactorEngine:
    def __init__(self, metrics: CodebaseMetrics):
        self.metrics = metrics
        self.recommendations = []
        self.auto_fixes = []
    
    # Formula 151: φ‑Loop Unrolling Factor
    def check_loop_unrolling(self, func: FunctionInfo) -> Optional[str]:
        # Detect hardcoded unroll factors (4,8,16)
        content = self._get_function_content(func)
        unroll_pattern = re.compile(r'#pragma\s+unroll\s+(\d+)|for\s*\([^;]*;\s*\w+\s*<\s*(\d+)\s*;[^)]*\)')
        for match in unroll_pattern.finditer(content):
            val = int(match.group(1) or match.group(2))
            if val not in FIB:
                optimal = min(FIB, key=lambda x: abs(x - PHI * val))
                return f"Loop unroll factor {val} should be Fibonacci {optimal} (Formula 151)"
        return None
    
    # Formula 152: φ‑Tile Size Heuristic
    def check_tile_size(self, func: FunctionInfo) -> Optional[str]:
        if 'MegaMoE' in func.name or 'grouped_gemm' in func.name:
            content = self._get_function_content(func)
            # Look for tile size definitions
            tile_pattern = re.compile(r'(?:constexpr\s+)?int\s+(?:TILE|BLOCK)_(\w+)\s*=\s*(\d+)')
            for match in tile_pattern.finditer(content):
                val = int(match.group(2))
                if val not in FIB:
                    optimal = min(FIB, key=lambda x: abs(x - val))
                    return f"Tile size {val} should be Fibonacci {optimal} (Formula 152)"
        return None
    
    # Formula 153: φ‑Expert Grouping Function
    def check_expert_grouping(self, func: FunctionInfo) -> Optional[str]:
        if 'expert' in func.name.lower() and 'group' in func.name.lower():
            content = self._get_function_content(func)
            # Look for manual grouping logic
            if 'expert_idx /' in content and '%' not in content:
                return "Consider φ‑logarithmic expert grouping: group = floor(log_phi(expert_idx)) (Formula 153)"
        return None
    
    # Formula 154: φ‑Index Hash for FP4 Lookups
    def check_fp4_indexer(self, func: FunctionInfo) -> Optional[str]:
        if 'FP4' in func.name or 'Indexer' in func.name:
            content = self._get_function_content(func)
            if 'hash' in content and 'phi' not in content.lower():
                return "Use φ‑multiplicative hash for index mapping: h(k) = (k * φ) mod 1 (Formula 154)"
        return None
    
    # Formula 155: φ‑Kernel Fusion Decision
    def check_kernel_fusion(self) -> Optional[str]:
        # Heuristic: look for adjacent kernels that could be fused
        kernels = [f for f in self.metrics.file_infos 
                   for f in f.functions if 'kernel' in f.name.lower()]
        if len(kernels) > 2 * PHI:
            return "Consider φ‑threshold for kernel fusion: fuse if arithmetic intensity > φ (Formula 155)"
        return None
    
    # Formula 156: φ‑Template Depth Limit
    def check_template_depth(self, func: FunctionInfo) -> Optional[str]:
        if func.template_depth > 4:
            return f"Template depth {func.template_depth} exceeds φ³ limit (4). Simplify or use JIT. (Formula 156)"
        return None
    
    # Formula 157: φ‑Simplification of Warp Scheduling
    def check_warp_scheduling(self, func: FunctionInfo) -> Optional[str]:
        if 'warp' in func.name.lower() or 'blockDim' in self._get_function_content(func):
            content = self._get_function_content(func)
            if 'occupancy' in content or 'cudaOccupancyMaxPotentialBlockSize' in content:
                return "Simplify warp allocation: active_warps = floor(φ * total_warps) (Formula 157)"
        return None
    
    # Formula 158: φ‑Abstraction Layer Count
    def check_abstraction_layers(self) -> Optional[str]:
        # Count directory depth and namespace nesting
        dirs = set()
        for fi in self.metrics.file_infos:
            parts = Path(fi.path).parts
            for i in range(len(parts)):
                dirs.add(os.path.join(*parts[:i+1]))
        total_dirs = len(dirs)
        optimal = int(math.log(self.metrics.total_lines, PHI))
        if total_dirs > optimal * 1.5:
            return f"Abstraction layers ({total_dirs}) exceed φ‑optimal ({optimal}). Consider flattening. (Formula 158)"
        return None
    
    # Formula 159: φ‑API Surface Reduction
    def check_api_surface(self) -> Optional[str]:
        pub = self.metrics.public_functions
        intern = self.metrics.internal_functions
        if intern > 0:
            ratio = pub / intern
            target = PHI_INV ** 2  # ≈ 0.382
            if abs(ratio - target) > 0.2:
                return f"Public/Internal ratio {ratio:.2f} deviates from φ²⁻ (0.382). (Formula 159)"
        return None
    
    # Formula 160: φ‑Compile‑Time Switch Elimination
    def check_compile_switches(self, func: FunctionInfo) -> Optional[str]:
        content = self._get_function_content(func)
        ifdef_count = content.count('#ifdef') + content.count('#ifndef')
        if constexpr_count = content.count('if constexpr')
        if ifdef_count > 2 and if constexpr_count == 0:
            return f"Replace {ifdef_count} #ifdef with if constexpr using φ‑threshold (Formula 160)"
        return None
    
    # Formula 161: φ‑Error Handling Simplification
    def check_error_handling(self, func: FunctionInfo) -> Optional[str]:
        content = self._get_function_content(func)
        if 'TOLERANCE' in content or 'epsilon' in content or 'assert' in content:
            if 'phi' not in content.lower():
                return "Use φ‑scaled assertion: check if |x| > φ⁻ᵏ * tolerance (Formula 161)"
        return None
    
    # Formula 162: φ‑Logging Verbosity
    def check_logging(self) -> Optional[str]:
        # Look for logging macro usage
        for fi in self.metrics.file_infos:
            for macro, _ in fi.macros:
                if 'LOG' in macro:
                    return "Set logging verbosity to Fibonacci numbers (1,2,3,5,8,13...) (Formula 162)"
        return None
    
    # Formula 163: φ‑Test Case Reduction
    def check_test_cases(self) -> Optional[str]:
        test_files = [fi for fi in self.metrics.file_infos if 'test' in fi.path.lower()]
        if test_files:
            total_test_lines = sum(fi.lines for fi in test_files)
            target_lines = int(total_test_lines * (PHI_INV ** 3))  # 23.6%
            return f"Reduce test cases to {target_lines} lines (φ⁻³) using φ‑sparse sampling. (Formula 163)"
        return None
    
    # Formula 164: φ‑Dependency Graph Pruning
    def check_include_pruning(self) -> List[Tuple[str, str, str]]:
        """Return list of (file, include, reason) for pruning."""
        pruning = []
        include_graph = self.metrics.include_graph
        for file, includes in include_graph.items():
            for inc in includes:
                # Compute topological distance? Simplified: prune if not Fibonacci distance
                pruning.append((file, inc, "Consider pruning includes at non‑Fibonacci distances (Formula 164)"))
        return pruning[:5]  # limit
    
    # Formula 165: φ‑Macro Elimination Rule
    def check_macros(self, fi: FileInfo) -> List[str]:
        recs = []
        for macro, line in fi.macros:
            # Estimate complexity (heuristic: length)
            if len(macro) > 20:
                recs.append(f"Macro '{macro}' at {fi.path}:{line} exceeds complexity; replace with constexpr (Formula 165)")
        return recs
    
    # Formula 166: φ‑Branch Predictor Hint
    def check_branch_hints(self, func: FunctionInfo) -> Optional[str]:
        content = self._get_function_content(func)
        if '__builtin_expect' in content or '[[likely]]' in content:
            # Check if used correctly
            if '[[likely]]' in content and 'if' in content:
                return "Use [[likely]] only when branch probability > 61.8% (Formula 166)"
        return None
    
    # Formula 167: φ‑Memory Pool Size
    def check_memory_pools(self, func: FunctionInfo) -> Optional[str]:
        if 'pool' in func.name.lower() or 'alloc' in func.name.lower():
            content = self._get_function_content(func)
            size_pattern = re.compile(r'(?:pool|block)_size\s*=\s*(\d+)')
            for match in size_pattern.finditer(content):
                val = int(match.group(1))
                if val not in FIB and val > 100:
                    optimal = min(FIB, key=lambda x: abs(x * 1024 - val))
                    return f"Memory pool size {val} should be Fibonacci multiple (e.g., {optimal}) (Formula 167)"
        return None
    
    # Formula 168: φ‑JIT Cache Key Simplification
    def check_jit_cache(self, func: FunctionInfo) -> Optional[str]:
        if 'JIT' in func.name or 'Cache' in func.name:
            content = self._get_function_content(func)
            if 'std::hash' in content or 'boost::hash' in content:
                return "Simplify JIT cache key: Hash(KernelName ⊕ (Params mod φ)) (Formula 168)"
        return None
    
    # Formula 169: φ‑Code Duplication Threshold
    def check_duplication(self) -> Optional[str]:
        # Very simplified duplication check: look for identical function signatures
        signatures = Counter()
        for fi in self.metrics.file_infos:
            for func in fi.functions:
                sig = f"{func.name}({','.join(func.params)})"
                signatures[sig] += 1
        duplicated = sum(1 for v in signatures.values() if v > 1)
        dup_ratio = duplicated / max(1, len(signatures))
        if dup_ratio > PHI_INV ** 2:  # 0.382
            return f"Code duplication ratio {dup_ratio:.2f} exceeds φ⁻² (0.382). Refactor. (Formula 169)"
        return None
    
    # Formula 170: Grand φ‑Simplicity Score
    def compute_simplicity_score(self) -> float:
        """Performance per line of code, normalized by φ."""
        # Since we don't have actual performance, use a proxy: functions per 1K lines
        func_density = self.metrics.total_functions / (self.metrics.total_lines / 1000)
        # Ideal density is around φ
        score = func_density / PHI
        return min(1.0, score)
    
    # ------------------------------------------------------------------------
    # Helper: Get function content
    def _get_function_content(self, func: FunctionInfo) -> str:
        try:
            with open(func.file_path, 'r') as f:
                lines = f.readlines()
            return ''.join(lines[func.line_start-1:func.line_end])
        except:
            return ""
    
    # ------------------------------------------------------------------------
    # Run all checks
    def analyze(self):
        for fi in self.metrics.file_infos:
            for func in fi.functions:
                for checker in [
                    self.check_loop_unrolling,
                    self.check_tile_size,
                    self.check_expert_grouping,
                    self.check_fp4_indexer,
                    self.check_template_depth,
                    self.check_warp_scheduling,
                    self.check_compile_switches,
                    self.check_error_handling,
                    self.check_branch_hints,
                    self.check_memory_pools,
                    self.check_jit_cache,
                ]:
                    rec = checker(func)
                    if rec:
                        self.recommendations.append((func.file_path, func.line_start, rec))
                
                # Macro checks per file (avoid duplicates)
                if not hasattr(self, '_macro_checked'):
                    for rec in self.check_macros(fi):
                        self.recommendations.append((fi.path, 0, rec))
                    setattr(self, '_macro_checked', True)
        
        # Global checks
        for checker in [
            self.check_kernel_fusion,
            self.check_abstraction_layers,
            self.check_api_surface,
            self.check_logging,
            self.check_test_cases,
            self.check_duplication,
        ]:
            rec = checker()
            if rec:
                self.recommendations.append(("", 0, rec))
        
        # Include pruning
        for file, inc, reason in self.check_include_pruning():
            self.recommendations.append((file, 0, f"Include '{inc}': {reason}"))
        
        return self.recommendations

# ============================================================================
# Main Analysis and Refactoring Tool
# ============================================================================
def build_codebase_metrics(root_path: str) -> CodebaseMetrics:
    parser = SimpleCppParser()
    file_infos = []
    total_lines = 0
    include_graph = defaultdict(set)
    
    extensions = {'.cpp', '.cxx', '.cc', '.cu', '.h', '.hpp', '.cuh'}
    for root, dirs, files in os.walk(root_path):
        # Skip build and third-party dirs
        dirs[:] = [d for d in dirs if d not in {'build', 'third_party', 'external', '.git'}]
        for file in files:
            if Path(file).suffix in extensions:
                full_path = os.path.join(root, file)
                try:
                    fi = parser.parse_file(full_path)
                    file_infos.append(fi)
                    total_lines += fi.lines
                    include_graph[full_path] = set(fi.includes)
                except Exception as e:
                    print(f"Warning: Could not parse {full_path}: {e}", file=sys.stderr)
    
    all_functions = [f for fi in file_infos for f in fi.functions]
    public_funcs = [f for f in all_functions if f.is_public]
    internal_funcs = [f for f in all_functions if not f.is_public]
    macro_count = sum(len(fi.macros) for fi in file_infos)
    template_count = sum(len(fi.templates) for fi in file_infos)
    avg_cyclomatic = sum(f.cyclomatic for f in all_functions) / max(1, len(all_functions))
    
    return CodebaseMetrics(
        total_lines=total_lines,
        total_files=len(file_infos),
        total_functions=len(all_functions),
        public_functions=len(public_funcs),
        internal_functions=len(internal_funcs),
        macro_count=macro_count,
        template_instantiations=template_count,
        avg_cyclomatic=avg_cyclomatic,
        include_graph=dict(include_graph),
        file_infos=file_infos
    )

def apply_auto_fixes(metrics: CodebaseMetrics, safe_mode: bool = True) -> List[str]:
    """Apply automatic refactorings that are safe."""
    fixes = []
    # Example: Fix macro to constexpr (if safe)
    for fi in metrics.file_infos:
        for macro, line in fi.macros:
            if macro.startswith('PHI_') or 'GOLDEN' in macro:
                continue  # keep our own macros
            # Very simple macro that is just a constant
            content = open(fi.path).readlines()[line-1]
            if re.match(r'^\s*#define\s+\w+\s+[\d.]+', content):
                if not safe_mode:
                    # Would replace with constexpr
                    fixes.append(f"Would replace macro '{macro}' with constexpr in {fi.path}:{line}")
    return fixes

# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="φ‑Coherent Complexity Reduction for C++/CUDA")
    parser.add_argument('--path', type=str, required=True, help='Root path of codebase')
    parser.add_argument('--analyze', action='store_true', help='Analyze and report recommendations')
    parser.add_argument('--apply', action='store_true', help='Apply automatic fixes (with --safe)')
    parser.add_argument('--safe', action='store_true', help='Only safe automatic fixes')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a valid directory", file=sys.stderr)
        sys.exit(1)
    
    print("φ‑Refactor: Analyzing codebase...")
    metrics = build_codebase_metrics(args.path)
    
    engine = PhiRefactorEngine(metrics)
    recommendations = engine.analyze()
    
    if args.json:
        output = {
            'metrics': {
                'total_lines': metrics.total_lines,
                'total_files': metrics.total_files,
                'total_functions': metrics.total_functions,
                'public_functions': metrics.public_functions,
                'internal_functions': metrics.internal_functions,
                'macro_count': metrics.macro_count,
                'avg_cyclomatic': metrics.avg_cyclomatic,
                'simplicity_score': engine.compute_simplicity_score()
            },
            'recommendations': [
                {'file': rec[0], 'line': rec[1], 'message': rec[2]}
                for rec in recommendations
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n=== φ‑Coherent Complexity Analysis ===\n")
        print(f"Total Lines: {metrics.total_lines}")
        print(f"Total Functions: {metrics.total_functions} ({metrics.public_functions} public, {metrics.internal_functions} internal)")
        print(f"Macros: {metrics.macro_count}")
        print(f"Template Instantiations: {metrics.template_instantiations}")
        print(f"Avg Cyclomatic Complexity: {metrics.avg_cyclomatic:.2f}")
        print(f"φ‑Simplicity Score: {engine.compute_simplicity_score():.3f} (target: 1.0)")
        print(f"\nOptimal Line Count (φ⁻²): {int(metrics.total_lines * PHI_INV**2)}")
        print(f"Current reduction opportunity: {metrics.total_lines - int(metrics.total_lines * PHI_INV**2)} lines")
        
        print("\n--- Recommendations ---")
        if recommendations:
            for file, line, msg in recommendations[:20]:
                if file:
                    print(f"  {file}:{line}: {msg}")
                else:
                    print(f"  [Global] {msg}")
            if len(recommendations) > 20:
                print(f"  ... and {len(recommendations)-20} more")
        else:
            print("  No recommendations. Codebase is φ‑coherent!")
    
    if args.apply:
        fixes = apply_auto_fixes(metrics, args.safe)
        print("\n--- Automatic Fixes ---")
        if fixes:
            for fix in fixes:
                print(f"  {fix}")
        else:
            print("  No automatic fixes available.")

if __name__ == "__main__":
    main()
