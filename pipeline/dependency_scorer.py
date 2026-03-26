"""AST-based import parsing and networkx dependency graph scoring."""

import ast
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from pipeline.config import SKIP_DIRS, SCORE_WEIGHTS

logger = logging.getLogger("pipeline.dependency_scorer")


@dataclass
class DependencyResult:
    dependency_score: float = 0.0
    internal_import_count: int = 0
    unique_internal_imports: int = 0
    graph_density: float = 0.0
    avg_in_degree: float = 0.0
    max_in_degree: int = 0
    num_connected_components: int = 0
    largest_component_fraction: float = 0.0


def _build_python_module_index(repo_path: str) -> dict[str, str]:
    """Map module names to file paths for all .py files in the repo."""
    module_index: dict[str, str] = {}
    repo = Path(repo_path)

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]

        for fname in files:
            if not fname.endswith(".py"):
                continue

            filepath = Path(root) / fname
            try:
                rel = filepath.relative_to(repo)
            except ValueError:
                continue

            # Convert file path to module name
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]  # strip .py

            if parts:
                module_name = ".".join(parts)
                module_index[module_name] = str(filepath)

    return module_index


def _extract_python_imports(filepath: str) -> list[str]:
    """Extract imported module names from a Python file using AST."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, ValueError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Handle relative imports
                if node.level > 0:
                    imports.append(f"__relative__.{node.module}")
                else:
                    imports.append(node.module)
            elif node.level > 0:
                # from . import something
                imports.append("__relative__")

    return imports


def _resolve_relative_import(
    importing_file: str, module_name: str, repo_path: str
) -> str | None:
    """Resolve a relative import to an absolute module name."""
    if not module_name.startswith("__relative__"):
        return module_name

    rel_module = module_name.replace("__relative__.", "").replace("__relative__", "")
    importing_dir = Path(importing_file).parent
    try:
        rel_to_repo = importing_dir.relative_to(repo_path)
    except ValueError:
        return None

    parent_parts = list(rel_to_repo.parts)
    if rel_module:
        candidate = ".".join(parent_parts + rel_module.split("."))
    else:
        candidate = ".".join(parent_parts) if parent_parts else None

    return candidate


def _is_internal_import(
    module_name: str, module_index: dict[str, str]
) -> bool:
    """Check if an import refers to an internal module."""
    # Direct match
    if module_name in module_index:
        return True
    # Check if it's a submodule of an internal package
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in module_index:
            return True
    return False


def _extract_js_ts_imports(filepath: str, repo_path: str) -> list[tuple[str, str]]:
    """Extract internal imports from JS/TS files using regex."""
    edges = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return edges

    # Match: import ... from './...', require('./...')
    patterns = [
        r"""(?:import|export)\s+.*?from\s+['"](\.[^'"]+)['"]""",
        r"""require\(\s*['"](\.[^'"]+)['"]\s*\)""",
    ]

    rel_path = os.path.relpath(filepath, repo_path)
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            target = match.group(1)
            # Normalize the import path
            source_dir = os.path.dirname(rel_path)
            resolved = os.path.normpath(os.path.join(source_dir, target))
            edges.append((rel_path, resolved))

    return edges


def _extract_c_cpp_includes(filepath: str, repo_path: str) -> list[tuple[str, str]]:
    """Extract internal #include "..." from C/C++ files."""
    edges = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return edges

    rel_path = os.path.relpath(filepath, repo_path)
    for match in re.finditer(r'#include\s+"([^"]+)"', content):
        target = match.group(1)
        source_dir = os.path.dirname(rel_path)
        resolved = os.path.normpath(os.path.join(source_dir, target))
        edges.append((rel_path, resolved))

    return edges


def _extract_rust_imports(filepath: str, repo_path: str) -> list[tuple[str, str]]:
    """Extract use crate:: imports from Rust files."""
    edges = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return edges

    rel_path = os.path.relpath(filepath, repo_path)
    for match in re.finditer(r"use\s+crate::(\w+)", content):
        target = match.group(1)
        edges.append((rel_path, target))

    return edges


def score_dependencies(repo_path: str) -> DependencyResult:
    """Build dependency graph and compute scoring metrics."""
    result = DependencyResult()
    G = nx.DiGraph()

    # Collect all source files
    py_files = []
    js_ts_files = []
    c_cpp_files = []
    rust_files = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fname in files:
            filepath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".py":
                py_files.append(filepath)
            elif ext in (".js", ".jsx", ".ts", ".tsx"):
                js_ts_files.append(filepath)
            elif ext in (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"):
                c_cpp_files.append(filepath)
            elif ext == ".rs":
                rust_files.append(filepath)

    # Determine primary language approach
    all_internal_imports = set()
    total_internal_imports = 0

    if py_files:
        # Python: AST-based analysis
        module_index = _build_python_module_index(repo_path)

        for filepath in py_files:
            rel_path = os.path.relpath(filepath, repo_path)
            G.add_node(rel_path)

            imports = _extract_python_imports(filepath)
            for imp in imports:
                resolved = imp
                if imp.startswith("__relative__"):
                    resolved = _resolve_relative_import(imp, imp, repo_path)
                    if not resolved:
                        continue

                if _is_internal_import(resolved, module_index):
                    # Find the target file
                    parts = resolved.split(".")
                    target_rel = None
                    for i in range(len(parts), 0, -1):
                        prefix = ".".join(parts[:i])
                        if prefix in module_index:
                            target_rel = os.path.relpath(
                                module_index[prefix], repo_path
                            )
                            break

                    if target_rel and target_rel != rel_path:
                        G.add_edge(rel_path, target_rel)
                        all_internal_imports.add((rel_path, target_rel))
                        total_internal_imports += 1

    # JS/TS imports
    for filepath in js_ts_files:
        rel_path = os.path.relpath(filepath, repo_path)
        G.add_node(rel_path)
        for source, target in _extract_js_ts_imports(filepath, repo_path):
            G.add_edge(source, target)
            all_internal_imports.add((source, target))
            total_internal_imports += 1

    # C/C++ includes
    for filepath in c_cpp_files:
        rel_path = os.path.relpath(filepath, repo_path)
        G.add_node(rel_path)
        for source, target in _extract_c_cpp_includes(filepath, repo_path):
            G.add_edge(source, target)
            all_internal_imports.add((source, target))
            total_internal_imports += 1

    # Rust imports
    for filepath in rust_files:
        rel_path = os.path.relpath(filepath, repo_path)
        G.add_node(rel_path)
        for source, target in _extract_rust_imports(filepath, repo_path):
            G.add_edge(source, target)
            all_internal_imports.add((source, target))
            total_internal_imports += 1

    result.internal_import_count = total_internal_imports
    result.unique_internal_imports = len(all_internal_imports)

    # Compute graph metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes < 2:
        result.dependency_score = 0.0
        return result

    result.graph_density = nx.density(G)

    in_degrees = [d for _, d in G.in_degree()]
    result.avg_in_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0.0
    result.max_in_degree = max(in_degrees) if in_degrees else 0

    # Connected components (on undirected view)
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    result.num_connected_components = len(components)

    if components:
        largest = max(len(c) for c in components)
        result.largest_component_fraction = largest / num_nodes
    else:
        result.largest_component_fraction = 0.0

    # Composite score
    # Normalize avg_in_degree by log(N) to make it scale-independent
    log_n = math.log(num_nodes) if num_nodes > 1 else 1.0
    normalized_avg_indeg = min(result.avg_in_degree / log_n, 1.0)

    # Edge-to-node ratio, capped at 1.0
    edge_node_ratio = min(num_edges / num_nodes, 1.0) if num_nodes > 0 else 0.0

    density = min(result.graph_density * 10, 1.0)  # scale up since density is usually very small

    result.dependency_score = (
        SCORE_WEIGHTS["density"] * density
        + SCORE_WEIGHTS["normalized_avg_indeg"] * normalized_avg_indeg
        + SCORE_WEIGHTS["largest_component_fraction"] * result.largest_component_fraction
        + SCORE_WEIGHTS["edge_node_ratio"] * edge_node_ratio
    )

    # Clamp to [0, 1]
    result.dependency_score = max(0.0, min(1.0, result.dependency_score))

    return result
