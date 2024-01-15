import numpy as np


class DensityMatrix:
    @staticmethod
    def trace(matrix):
        """
        Calculate the trace of a given matrix.

        Parameters:
            matrix (numpy.ndarray): The input matrix for which the trace needs to be calculated.

        Returns:
            float: The trace of the matrix.
        """
        return np.trace(matrix)

    def __init__(self, data):
        """
        Initializes the object with the given data.

        Parameters:
            data (array-like): The input data to initialize the object. It must be convertible to a numpy array.

        Raises:
            ValueError: If the input data cannot be converted to a numpy array or if it is not an array of complex numbers.

        Returns:
            None
        """
        try:
            array = np.asarray(data)
        except ValueError:
            raise ValueError("Input must be convertible to a numpy array.")

        if not np.iscomplexobj(array):
            try:
                array = array.astype(complex)
            except:
                raise ValueError("Input must be an array of complex numbers.")

        if array.ndim == 1:
            self.initialize_from_statevector(array)
        elif array.ndim == 2:
            self.initialize_from_matrix(array)
        else:
            raise ValueError("Input must be a 1D or 2D array.")

    def initialize_from_statevector(self, statevector):
        """
        Initializes the quantum circuit from a given state vector.

        Parameters:
            statevector (np.ndarray): The state vector to initialize the circuit from.

        Raises:
            ValueError: If the state vector is empty or if its dimensions are not a power of 2.

        Returns:
            None
        """
        if statevector.size == 0:
            raise ValueError("Statevector must not be empty.")
        if not self._is_power_of_two(statevector.size):
            raise ValueError("Statevector dimensions must be a power of 2.")
        self.initialize_from_matrix(np.outer(statevector, statevector.conj()))

    def initialize_from_matrix(self, matrix):
        """
        Initializes the density matrix from a given matrix.

        Parameters:
            matrix (ndarray): The matrix representing the density matrix.

        Raises:
            ValueError: If the matrix is empty, not square, dimensions are not a power of 2,
                        or if the matrix has a trace of zero.
            ValueError: If the matrix is not Hermitian.
            ValueError: If the matrix is not positive semidefinite.

        Returns:
            None
        """
        if matrix.size == 0:
            raise ValueError("Density matrices must not be empty.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Density matrices must be square.")
        if not self._is_power_of_two(matrix.shape[0]):
            raise ValueError(
                "Density matrices must have dimensions that are a power of 2."
            )
        if self.trace(matrix) == 0:
            raise ValueError("Cannot normalize a matrix with a trace of zero.")
        matrix = matrix / self.trace(matrix)
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Density matrices must be Hermitian.")
        if np.any(np.linalg.eigvalsh(matrix) < 0):
            raise ValueError("Density matrices must be positive semidefinite.")
        self.matrix = matrix

    def purity(self):
        """
        Calculates the purity of the density matrix.

        Returns:
            float: The purity value of the density matrix.

        Raises:
            ValueError: If the density matrix is not unitary.
        """
        if not hasattr(self, "_purity"):
            self._purity = DensityMatrix.trace(self.matrix @ self.matrix)
            if self._purity > 1:
                raise ValueError("The density matrix is not unitary.")
        return self._purity

    def is_pure(self):
        """
        Determines if the object is pure.

        Returns:
            bool: True if the object is pure, False otherwise.
        """
        return self.purity() == 1

    @staticmethod
    def _is_power_of_two(n):
        """
        Determines if a given number is a power of two.

        Args:
            n (int): The number to check.

        Returns:
            bool: True if the number is a power of two, False otherwise.
        """
        return (n & (n - 1)) == 0 and n != 0

    def __repr__(self):
        """
        Return a string representation of the DensityMatrix object.

        Parameters:
            self (DensityMatrix): The DensityMatrix object.

        Returns:
            str: A string representation of the DensityMatrix object.
        """
        rows = [" ".join(str(element) for element in row) for row in self.matrix]
        matrix_str = "\n".join(rows)
        return f"DensityMatrix(\n{matrix_str}\n)"
