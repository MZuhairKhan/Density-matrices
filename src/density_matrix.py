import numpy as np


class DensityMatrix:
    @staticmethod
    def trace(matrix):
        return np.trace(matrix)

    def __init__(self, data):
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
        if statevector.size == 0:
            raise ValueError("Statevector must not be empty.")
        if not self._is_power_of_two(statevector.size):
            raise ValueError("Statevector dimensions must be a power of 2.")
        self.initialize_from_matrix(np.outer(statevector, statevector.conj()))

    def initialize_from_matrix(self, matrix):
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
        if not hasattr(self, "_purity"):
            self._purity = DensityMatrix.trace(self.matrix @ self.matrix)
            if self._purity > 1:
                raise ValueError("The density matrix is not unitary.")
        return self._purity

    def is_pure(self):
        return self.purity() == 1

    @staticmethod
    def _is_power_of_two(n):
        return (n & (n - 1)) == 0 and n != 0

    def __repr__(self):
        rows = [" ".join(str(element) for element in row) for row in self.matrix]
        matrix_str = "\n".join(rows)
        return f"DensityMatrix(\n{matrix_str}\n)"
