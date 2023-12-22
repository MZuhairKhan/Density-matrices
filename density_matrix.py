import numpy as np


class DensityMatrix:
    @staticmethod
    def trace(matrix):
        return np.trace(matrix)

    @staticmethod
    def is_unitary(matrix):
        return DensityMatrix.trace(matrix) == 1

    def __init__(self, data):
        try:
            array = np.array(data)
        except TypeError:
            raise TypeError("Input must be iterable.")

        if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(
            array.dtype, complex
        ):
            raise ValueError("Input must be an array of complex numbers.")

        dimension = np.ndim(array)
        if dimension == 1:
            self.initialize_from_statevector(array)
        elif dimension == 2:
            self.initialize_from_matrix(array)
        else:
            raise ValueError("Input must be a 1D or 2D array.")

    def initialize_from_statevector(self, statevector):
        if statevector.shape[0] == 0:
            raise ValueError("Statevector must not be empty.")
        elif (statevector.shape[0] & (statevector.shape[0] - 1)) != 0:
            raise ValueError("Statevector dimensions must be a power of 2.")
        else:
            matrix = np.outer(statevector, statevector.conj())
            if DensityMatrix.is_unitary(matrix) == False:
                raise ValueError("Enter a valid statevector.")
            else:
                self.matrix = matrix

    def initialize_from_matrix(self, matrix):
        if matrix.shape[0] == 0:
            raise ValueError("Density matrices must not be empty.")
        elif matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Density matrices must be square.")
        elif (matrix.shape[0] & (matrix.shape[0] - 1)) != 0:
            raise ValueError(
                "Density matrices must have dimensions that are a power of 2."
            )
        elif DensityMatrix.is_unitary(matrix) == False:
            raise ValueError("Density matrices must be unitary.")
        else:
            self.matrix = matrix

    def purity(self):
        purity = DensityMatrix.trace(matrix * matrix)
        if purity <= 1:
            return purity
        else:
            raise ValueError("The density matrix is not unitary.")

    def is_pure(self):
        if self.purity() == 1:
            return True
        elif self.purity() < 1:
            return False

    def __repr__(self):
        rows = []
        for row in self.matrix:
            rows.append(" ".join(str(element) for element in row))

        matrix_str = "\n".join(rows)
        return f"Matrix(\n{matrix_str}\n)"
