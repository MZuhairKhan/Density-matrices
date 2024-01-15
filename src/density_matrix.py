from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy


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
            array: nd.ndarray = np.asarray(data, dtype=complex)
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

    def _is_power_of_two(self, n: int) -> bool:
        """
        Determines if a given number is a power of two.

        Parameters:
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
        sep: str = "  "
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

    def partial_trace(
        self, subsystems_to_trace_out: Union[int, Iterable]
    ) -> DensityMatrix:
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
            subsystems_to_trace_out: nd.ndarray[int] = np.asarray(
                subsystems_to_trace_out, dtype=int
            )
        except ValueError:
            raise ValueError(
                "The subsystems to trace out must be an integer or an iterable."
            )

        density_matrix: np.ndarray[complex] = self.matrix()

        for subsystem_index in subsystems_to_trace_out:
            dimensions: int = self._calculate_square_matrix_dimension(density_matrix)
            subsystem_index = subsystem_index - 1
            if dimensions <= subsystem_index or subsystem_index < 0:
                raise ValueError(
                    f"The subsystem index {subsystem_index} is out of bounds."
                )

            new_shape: List[int] = [2] * (2 * dimensions)
            matrix_reshaped: np.ndarray = density_matrix.reshape(new_shape)

            axis_to_sum: Tuple = (subsystem_index, subsystem_index + dimensions)
            partial_matrix: np.ndarray = np.trace(
                matrix_reshaped, axis1=axis_to_sum[0], axis2=axis_to_sum[1]
            )

            final_shape: List[int] = [2] * (2 * dimensions - 2)
            density_matrix: np.ndarray = partial_matrix.reshape(final_shape)

        return DensityMatrix(density_matrix)

    def eigensystem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and eigenvectors of the matrix.

        Returns:
            Tuple: A tuple containing the eigenvalues and eigenvectors of the matrix.
                   The eigenvalues are stored in a 1-D array, and the eigenvectors are
                   stored in a 2-D array where each column represents an eigenvector.
        """
        return np.linalg.eigh(self.matrix())

    def evolve(self, operator: nd.ndarray) -> "DensityMatrix":
        """
        Evolves the density matrix by applying an operator.

        Parameters:
            operator (ndarray): The operator to apply to the density matrix.

        Returns:
            DensityMatrix: The evolved density matrix.
        """
        return DensityMatrix(operator @ self.matrix() @ operator.conj().T)

    def measurement_probability(self, operator: nd.ndarray) -> float:
        """
        Calculate the measurement probability of a given operator.

        Parameters:
            operator (numpy.ndarray): The operator for which to calculate the
                                      measurement probability.

        Returns:
            float: The measurement probability of the operator.
        """
        return np.trace(self.matrix() @ operator)

    def fidelity(self, other) -> float:
        """
        Calculates the fidelity between two quantum states.

        Parameters:
            other (QuantumState): The other quantum state to compare against.

        Returns:
            float: The fidelity value between the two states.
        """
        sqrtm_self: np.ndarray = scipy.linalg.sqrtm(self.matrix())
        product_matrix: np.ndarray = sqrtm_self @ other._matrix @ sqrtm_self
        return np.trace(scipy.linalg.sqrtm(product_matrix)) ** 2

    def von_neumann_entropy(self) -> float:
        """
        Calculates the von Neumann entropy of a quantum system.

        Returns:
            float: The von Neumann entropy of the system.

        Raises:
            ValueError: If the eigenvalues of the system matrix are not positive.
        """
        eigenvalues: np.ndarray = np.linalg.eigvalsh(self.matrix())
        return -np.sum(eigenvalues * np.log2(eigenvalues, where=eigenvalues > 0))
