"""Graph representation utilities for adjacency matrices."""
from __future__ import annotations
import numpy as np

def validate_adjacency(matrix: np.ndarray) -> bool:
    """Check that matrix is a valid adjacency matrix (square, symmetric, zero diagonal, binary)."""
    if matrix.ndim != 2:
        return False
    n, m = matrix.shape
    if n != m:
        return False
    if not np.all(np.diag(matrix) == 0):
        return False
    if not np.array_equal(matrix, matrix.T):
        return False
    if not np.all((matrix == 0) | (matrix == 1)):
        return False
    return True

def complement(matrix: np.ndarray) -> np.ndarray:
    """Return the complement graph: flip edges, keep diagonal zero."""
    c = 1 - matrix
    np.fill_diagonal(c, 0)
    return c.astype(np.int8)

def to_edge_list(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return sorted list of edges (i, j) where i < j and matrix[i][j] == 1."""
    edges = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] == 1:
                edges.append((i, j))
    return edges

def from_edge_list(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    """Build an n x n adjacency matrix from a list of edges."""
    matrix = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        if i < 0 or j < 0 or i >= n or j >= n:
            raise ValueError(f"Edge ({i}, {j}) out of bounds for n={n}")
        if i == j:
            raise ValueError(f"Self-loop ({i}, {i}) not allowed")
        matrix[i, j] = 1
        matrix[j, i] = 1
    return matrix
