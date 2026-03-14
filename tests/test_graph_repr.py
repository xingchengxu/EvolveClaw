import numpy as np
import pytest
from evolveclaw_ramsey.ramsey.graph_repr import validate_adjacency, complement, to_edge_list, from_edge_list

def test_validate_adjacency_valid():
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    assert validate_adjacency(m) is True

def test_validate_adjacency_not_square():
    m = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int8)
    assert validate_adjacency(m) is False

def test_validate_adjacency_nonzero_diagonal():
    m = np.array([[1, 1], [1, 0]], dtype=np.int8)
    assert validate_adjacency(m) is False

def test_validate_adjacency_not_symmetric():
    m = np.array([[0, 1], [0, 0]], dtype=np.int8)
    assert validate_adjacency(m) is False

def test_complement():
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    c = complement(m)
    expected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(c, expected)

def test_edge_list_roundtrip():
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    edges = to_edge_list(m)
    assert set(edges) == {(0, 1), (1, 2)}
    m2 = from_edge_list(edges, 3)
    np.testing.assert_array_equal(m, m2)

def test_from_edge_list_out_of_bounds():
    with pytest.raises(ValueError, match="out of bounds"):
        from_edge_list([(0, 10)], n=5)

def test_from_edge_list_self_loop():
    with pytest.raises(ValueError, match="Self-loop"):
        from_edge_list([(2, 2)], n=5)

def test_from_edge_list_negative_index():
    with pytest.raises(ValueError, match="out of bounds"):
        from_edge_list([(-1, 0)], n=5)
