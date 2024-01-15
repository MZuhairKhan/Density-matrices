import unittest

import numpy as np

from src.density_matrix import DensityMatrix


class TestDensityMatrix(unittest.TestCase):
    def test_initialize_from_valid_statevector(self):
        """
        Initialize the DensityMatrix object from a valid statevector and test if the resulting density matrix is correct.

        Parameters:
            self (TestDensityMatrix): The test case object.

        Returns:
            None
        """
        statevector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        density_matrix = DensityMatrix(statevector)
        expected_matrix = np.outer(statevector, statevector.conj())
        self.assertTrue(np.allclose(density_matrix.matrix, expected_matrix))

    def test_initialize_from_invalid_statevector(self):
        """
        Test case for initializing a DensityMatrix object from an invalid state vector.

        Args:
            self: The test case object.

        Returns:
            None.
        """
        statevector = np.array([0, 0])  # Not a valid state vector as it has trace 0.
        with self.assertRaises(ValueError):
            DensityMatrix(statevector)

    def test_initialize_from_valid_matrix(self):
        """
        Test the initialization of the DensityMatrix class from a valid matrix.

        Parameters:
            self (TestCase): The current test case.

        Returns:
            None
        """
        matrix = np.array([[1 / 2, 0], [0, 1 / 2]])
        density_matrix = DensityMatrix(matrix)
        self.assertTrue(np.allclose(density_matrix.matrix, matrix))

    def test_initialize_from_invalid_matrix(self):
        """
        Initializes a `DensityMatrix` object from an invalid matrix.

        This function takes in a matrix that is not a valid density matrix and attempts to initialize a `DensityMatrix` object with it. It then asserts that a `ValueError` is raised.

        Parameters:
            self (TestCase): The current test case.

        Returns:
            None

        Raises:
            ValueError: If the matrix is not a valid density matrix.
        """
        matrix = np.array([[0, 0], [0, 0]])  # Not a valid density matrix
        with self.assertRaises(ValueError):
            DensityMatrix(matrix)

    def test_purity_of_pure_state(self):
        """
        Test the purity of a pure state.

        This function calculates the purity of a pure state by creating a density matrix from a given state vector.
        It then asserts that the calculated purity is equal to 1.

        Parameters:
        - self: The object instance.

        Returns:
        - None
        """
        statevector = np.array([1, 0])
        density_matrix = DensityMatrix(statevector)
        self.assertEqual(density_matrix.purity(), 1)

    def test_purity_of_mixed_state(self):
        """
        Test the purity of a mixed state.

        This function creates a mixed state represented by a density matrix and
        checks if its purity is less than 1. The purity of a state is a measure
        of its mixedness, with a value of 1 indicating a pure state and a value
        less than 1 indicating a mixed state.

        Parameters:
            self (TestCase): The current test case.

        Returns:
            None
        """
        mixed_matrix = np.array([[0.5, 0], [0, 0.5]])
        density_matrix = DensityMatrix(mixed_matrix)
        self.assertLess(density_matrix.purity(), 1)


if __name__ == "__main__":
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output="test-reports"))
