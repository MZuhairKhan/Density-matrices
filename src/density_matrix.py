from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class DensityMatrix:
    def __init__(self, data) -> None:
        """
        Initializes the object with the given data.

        Parameters:
            data (array-like):  The input data to initialize the object.
                                It must be convertible to a numpy array.

        Raises:
            ValueError: If the input data cannot be converted to a numpy array or if
                        it is not an array of complex numbers.
        """
        try:
            array = np.asarray(data, dtype=complex)
        except ValueError:
            raise ValueError(
                "Input must be convertible to a numpy array of complex numbers."
            )

        if array.ndim not in {1, 2}:
            raise ValueError(
                f"Input must be a 1D or 2D array, got {array.ndim}D array."
            )

        if array.ndim == 1:
            self._initialize_from_statevector(array)
        elif array.ndim == 2:
            self._initialize_from_matrix(array)

    def _initialize_from_statevector(self, statevector: np.ndarray) -> None:
        """
        Initializes the quantum circuit from a given state vector.

        Parameters:
            statevector (np.ndarray): The state vector to initialize the circuit from.

        Raises:
            ValueError: If the state vector is empty.
        """
        if statevector.size == 0:
            raise ValueError("Statevector must not be empty.")
        self._initialize_from_matrix(np.outer(statevector, statevector.conj()))

    def _initialize_from_matrix(self, matrix: np.ndarray) -> None:
        """
        Initialize the object from a given matrix.

        Parameters:
            matrix (np.ndarray): The matrix to initialize from.
        """
        self._validate_matrix(matrix)
        self._matrix = matrix / np.trace(matrix)
        self._dimension = self._calculate_square_matrix_dimension(matrix)
        self._purity = np.trace(self._matrix @ self._matrix)

    def _validate_matrix(self, matrix: np.ndarray) -> None:
        """
        Validates a given matrix to ensure it meets certain criteria.

        Parameters:
            matrix (np.ndarray): The matrix to be validated.

        Raises:
            ValueError: If the matrix is empty or not square.
            ValueError: If the matrix is not Hermitian.
            ValueError: If the matrix is not positive semidefinite.
            ValueError: If the matrix has a trace of zero.
        """
        if matrix.size == 0:
            raise ValueError("Density matrices must not be empty.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Density matrices must be square.")
        self._validate_square_matrix(matrix)
        self._validate_hermitian(matrix)
        self._validate_positive_semidefinite(matrix)

    def _validate_square_matrix(self, matrix: np.ndarray) -> None:
        """
        Validates the dimensions of a square matrix.

        Parameters:
            matrix (np.ndarray): The square matrix to validate.

        Raises:
            ValueError: If the dimensions of the matrix are not a power of 2
                        or if the matrix has a trace of zero.
        """
        if not self._is_power_of_two(matrix.shape[0]):
            raise ValueError(
                "Density matrices must have dimensions that are a power of 2, "
                + f"got {matrix.shape[0]}."
            )
        if np.trace(matrix) == 0:
            raise ValueError("Cannot normalize a matrix with a trace of zero.")

    def _validate_hermitian(self, matrix: np.ndarray) -> None:
        """
        Validate whether a given matrix is Hermitian.

        Parameters:
            matrix (np.ndarray): The matrix to be validated.

        Raises:
            ValueError: If the matrix is not Hermitian.
        """
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Density matrices must be Hermitian.")

    def _validate_positive_semidefinite(self, matrix: np.ndarray) -> None:
        """
        Validate whether a given matrix is positive semidefinite.

        Parameters:
            matrix (np.ndarray): The matrix to be validated.

        Raises:
            ValueError: If the matrix is not positive semidefinite.
        """
        if np.any(np.linalg.eigvalsh(matrix) < 0):
            raise ValueError("Density matrices must be positive semidefinite.")

    def _calculate_square_matrix_dimension(self, matrix: np.ndarray) -> int:
        """
        Quickly calculates the dimension of a square matrix if the size is known
        to be a power of 2.

        Parameters:
            matrix (np.ndarray): The square matrix for which the dimension needs
                                 to be calculated.

        Returns:
            int: The dimension of the square matrix.
        """
        return matrix.shape[0].bit_length() - 1

    def purity(self) -> float:
        """
        Calculates the purity of the density matrix.

        Returns:
            float: The purity value of the density matrix.
        """
        return self._purity

    def is_pure(self) -> bool:
        """
        Determines if the object is pure.

        Returns:
            bool: True if the object is pure, False otherwise.
        """
        return np.isclose(self.purity(), 1)

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """
        Determines if a given number is a power of two.

        Args:
            n (int): The number to check.

        Returns:
            bool: True if the number is a power of two, False otherwise.
        """
        return (n & (n - 1)) == 0 and n != 0

    def __repr__(self) -> str:
        """
        Return a string representation of the DensityMatrix object.

        Returns:
            str: A string representation of the DensityMatrix object.
        """
        sep = "  "
        rows = [
            sep.join("{0.real:.1f}+{0.imag:.1f}j".format(element) for element in row)
            for row in self.matrix()
        ]
        matrix_str = ("\n" + sep).join(rows)
        return "DensityMatrix(\n" + sep + f"{matrix_str}\n)"

    def dimension(self) -> int:
        """
        Return the number of dimensions of the DensityMatrix object.

        Returns:
            int: The number of dimensions of the DensityMatrix object.
        """
        return self._dimension

    def matrix(self) -> np.ndarray:
        """
        Return the matrix of the DensityMatrix object.

        Returns:
            np.ndarray: The matrix of the DensityMatrix object.
        """
        return self._matrix

    def partial_trace(self, subsystems_to_trace_out) -> DensityMatrix:
        """
        Calculates the partial trace of the density matrix with respect to the given
        subsystems. The subsystems are either given as an integer or as an iterable
        of integers of the subsystems.

        Parameters:
            subsystems_to_trace_out (int or iterable):
                The subsystem indices to trace out.

        Returns:
            DensityMatrix: The density matrix after performing the partial trace.
        """
        if isinstance(subsystems_to_trace_out, int):
            subsystems_to_trace_out = [subsystems_to_trace_out]

        try:
            subsystems_to_trace_out = np.asarray(subsystems_to_trace_out, dtype=int)
        except ValueError:
            raise ValueError(
                "The subsystems to trace out must be an integer or an iterable."
            )

        density_matrix = self.matrix()

        for subsystem_index in subsystems_to_trace_out:
            dimensions = self._calculate_square_matrix_dimension(density_matrix)
            subsystem_index = subsystem_index - 1
            if dimensions <= subsystem_index or subsystem_index < 0:
                raise ValueError("The subsystem index is out of bounds.")

            new_shape = [2] * (2 * dimensions)
            matrix_reshaped = density_matrix.reshape(new_shape)

            axis_to_sum = (subsystem_index, subsystem_index + dimensions)
            partial_matrix = np.trace(
                matrix_reshaped, axis1=axis_to_sum[0], axis2=axis_to_sum[1]
            )

            final_shape = [2] * (2 * dimensions - 2)
            density_matrix = partial_matrix.reshape(final_shape)

        return DensityMatrix(density_matrix)
