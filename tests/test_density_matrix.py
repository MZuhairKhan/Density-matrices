import unittest

import numpy as np

from src.density_matrix import DensityMatrix


class TestDensityMatrix(unittest.TestCase):
    def test_initialize_from_valid_statevector(self):
        statevector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        density_matrix = DensityMatrix(statevector)
        expected_matrix = np.outer(statevector, statevector.conj())
        self.assertTrue(np.allclose(density_matrix.matrix, expected_matrix))

    def test_initialize_from_invalid_statevector(self):
        statevector = np.array(
            [0,0]
        )  # Not a valid state vector as it has trace 0.
        with self.assertRaises(ValueError):
            DensityMatrix(statevector)
    
    def test_initialize_from_valid_matrix(self):
        matrix = np.array([[1/2, 0], [0, 1/2]])
        density_matrix = DensityMatrix(matrix)
        self.assertTrue(np.allclose(density_matrix.matrix, matrix))
    
    def test_initialize_from_invalid_matrix(self):
        matrix = np.array([[0, 0], [0, 0]])  # Not a valid density matrix
        with self.assertRaises(ValueError):
            DensityMatrix(matrix)

    def test_purity_of_pure_state(self):
        statevector = np.array([1, 0])
        density_matrix = DensityMatrix(statevector)
        self.assertEqual(density_matrix.purity(), 1)

    def test_purity_of_mixed_state(self):
        mixed_matrix = np.array([[0.5, 0], [0, 0.5]])
        density_matrix = DensityMatrix(mixed_matrix)
        self.assertLess(density_matrix.purity(), 1)


if __name__ == "__main__":
    unittest.main()
