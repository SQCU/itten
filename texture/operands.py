"""
Operand input classes for texture synthesis.

Operands are modulation signals that influence the spectral etch.
They're typically simpler than carriers.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class OperandInput:
    """
    Base class for operand patterns (modulation signals).

    Operands are simpler than carriers - they typically don't need
    complex transform chains, just basic generation.
    """

    def __init__(self, size: int):
        """
        Initialize operand with target size.

        Args:
            size: Output size (square image: size x size)
        """
        self.size = size

    def render(self) -> np.ndarray:
        """
        Generate the operand pattern.

        Returns:
            2D numpy array (size, size) with values in [0, 1]
        """
        return self._generate()

    def _generate(self) -> np.ndarray:
        """
        Generate the operand pattern.

        Subclasses must implement this.

        Returns:
            2D numpy array (size, size)
        """
        raise NotImplementedError("Subclasses must implement _generate()")


class CheckerboardOperand(OperandInput):
    """Checkerboard operand pattern."""

    def __init__(self, size: int = 64, cells: int = 8):
        super().__init__(size)
        self.cells = cells

    def _generate(self) -> np.ndarray:
        from .patterns import generate_checkerboard
        tile_size = self.size // self.cells
        if tile_size < 1:
            tile_size = 1
        return generate_checkerboard(self.size, tile_size).astype(np.float32)


class NoiseOperand(OperandInput):
    """Random noise operand."""

    def __init__(self, size: int = 64, seed: int = 42, smooth: bool = True):
        super().__init__(size)
        self.seed = seed
        self.smooth = smooth

    def _generate(self) -> np.ndarray:
        np.random.seed(self.seed)
        noise = np.random.rand(self.size, self.size).astype(np.float32)

        if self.smooth:
            noise = gaussian_filter(noise, sigma=2)
            # Renormalize
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)

        return noise


class SolidOperand(OperandInput):
    """Solid (constant) operand."""

    def __init__(self, size: int = 64, value: float = 1.0):
        super().__init__(size)
        self.value = value

    def _generate(self) -> np.ndarray:
        return np.full((self.size, self.size), self.value, dtype=np.float32)


class GradientOperand(OperandInput):
    """Gradient operand pattern."""

    def __init__(self, size: int = 64, direction: str = 'horizontal'):
        super().__init__(size)
        self.direction = direction

    def _generate(self) -> np.ndarray:
        from .patterns import generate_gradient
        return generate_gradient(self.size, self.direction)


class CircleOperand(OperandInput):
    """Circle/disk operand pattern."""

    def __init__(self, size: int = 64, radius: float = 0.4, softness: float = 0.1):
        super().__init__(size)
        self.radius = radius
        self.softness = softness

    def _generate(self) -> np.ndarray:
        y, x = np.ogrid[:self.size, :self.size]
        cx, cy = self.size / 2, self.size / 2

        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / self.size

        if self.softness > 0:
            pattern = 1.0 - np.clip((dist - self.radius) / self.softness, 0, 1)
        else:
            pattern = (dist <= self.radius).astype(np.float32)

        return pattern.astype(np.float32)


class ArrayOperand(OperandInput):
    """Operand from existing numpy array."""

    def __init__(self, array: np.ndarray):
        """
        Args:
            array: 2D numpy array (will be normalized to [0, 1])
        """
        if array.ndim != 2:
            raise ValueError(f"Array must be 2D, got shape {array.shape}")
        size = max(array.shape)
        super().__init__(size)
        self._array = array.astype(np.float32)
        if self._array.max() > 1.0:
            self._array = self._array / 255.0

    def _generate(self) -> np.ndarray:
        # Resize if not square
        if self._array.shape[0] != self._array.shape[1]:
            from scipy.ndimage import zoom
            target = max(self._array.shape)
            factors = (target / self._array.shape[0], target / self._array.shape[1])
            return zoom(self._array, factors, order=1)
        return self._array.copy()
