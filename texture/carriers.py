"""
Carrier input classes for texture synthesis.

Carriers define the structure that guides spectral diffusion.
They support chainable transforms (rotate, stretch, translate).
"""

import numpy as np
from typing import List, Tuple
from scipy.ndimage import rotate as nd_rotate, zoom, shift


class CarrierInput:
    """
    Base class for carrier patterns with chainable transforms.

    Usage:
        carrier = AmongusCarrier(128).rotate(45).stretch(2, 1)
        img = carrier.render()
    """

    def __init__(self, size: int):
        """
        Initialize carrier with target size.

        Args:
            size: Output size (square image: size x size)
        """
        self.size = size
        self._transforms: List[Tuple] = []

    def rotate(self, degrees: float) -> 'CarrierInput':
        """
        Add rotation transform to the chain.

        Args:
            degrees: Rotation angle in degrees (counter-clockwise positive)

        Returns:
            Self for chaining
        """
        self._transforms.append(('rotate', degrees))
        return self

    def stretch(self, sx: float, sy: float) -> 'CarrierInput':
        """
        Add stretch/scale transform to the chain.

        Args:
            sx: Horizontal scale factor
            sy: Vertical scale factor

        Returns:
            Self for chaining
        """
        self._transforms.append(('stretch', sx, sy))
        return self

    def translate(self, dx: int, dy: int) -> 'CarrierInput':
        """
        Add translation transform to the chain.

        Args:
            dx: Horizontal offset (pixels)
            dy: Vertical offset (pixels)

        Returns:
            Self for chaining
        """
        self._transforms.append(('translate', dx, dy))
        return self

    def render(self) -> np.ndarray:
        """
        Generate base pattern and apply all queued transforms.

        Returns:
            2D numpy array (H, W) with values in [0, 1]
        """
        img = self._generate_base()
        for transform in self._transforms:
            img = self._apply_transform(img, transform)
        return img

    def _generate_base(self) -> np.ndarray:
        """
        Generate the base pattern before transforms.

        Subclasses must implement this.

        Returns:
            2D numpy array (size, size)
        """
        raise NotImplementedError("Subclasses must implement _generate_base()")

    def _apply_transform(self, img: np.ndarray, transform: Tuple) -> np.ndarray:
        """Apply a single transform to the image."""
        if transform[0] == 'rotate':
            return nd_rotate(img, transform[1], reshape=False, mode='wrap', order=1)

        elif transform[0] == 'stretch':
            sx, sy = transform[1], transform[2]
            zoomed = zoom(img, (sy, sx), order=1, mode='nearest')
            return self._crop_or_pad(zoomed, self.size)

        elif transform[0] == 'translate':
            dx, dy = transform[1], transform[2]
            return shift(img, (dy, dx), mode='wrap', order=1)

        return img

    def _crop_or_pad(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """Crop or pad image to target size, centered."""
        h, w = img.shape
        result = np.zeros((target_size, target_size), dtype=img.dtype)

        src_y_start = max(0, (h - target_size) // 2)
        src_x_start = max(0, (w - target_size) // 2)
        dst_y_start = max(0, (target_size - h) // 2)
        dst_x_start = max(0, (target_size - w) // 2)

        copy_h = min(h, target_size)
        copy_w = min(w, target_size)

        if h < target_size:
            src_y_start = 0
        if w < target_size:
            src_x_start = 0

        result[dst_y_start:dst_y_start + copy_h,
               dst_x_start:dst_x_start + copy_w] = \
            img[src_y_start:src_y_start + copy_h,
                src_x_start:src_x_start + copy_w]

        return result

    def reset_transforms(self) -> 'CarrierInput':
        """Clear all queued transforms."""
        self._transforms = []
        return self


class AmongusCarrier(CarrierInput):
    """Among Us crewmate silhouette pattern."""

    def __init__(self, size: int = 64):
        super().__init__(size)

    def _generate_base(self) -> np.ndarray:
        from .patterns import generate_amongus
        return generate_amongus(self.size).astype(np.float32)


class CheckerboardCarrier(CarrierInput):
    """Classic checkerboard pattern."""

    def __init__(self, size: int = 64, cells: int = 8):
        super().__init__(size)
        self.cells = cells

    def _generate_base(self) -> np.ndarray:
        from .patterns import generate_checkerboard
        tile_size = self.size // self.cells
        if tile_size < 1:
            tile_size = 1
        return generate_checkerboard(self.size, tile_size).astype(np.float32)


class NoiseCarrier(CarrierInput):
    """Multi-octave noise pattern."""

    def __init__(self, size: int = 64, scale: float = 4.0, seed: int = 42, octaves: int = 4):
        super().__init__(size)
        self.scale = scale
        self.seed = seed
        self.octaves = octaves

    def _generate_base(self) -> np.ndarray:
        from .patterns import generate_noise
        return generate_noise(self.size, self.scale, self.seed, self.octaves)


class DragonCurveCarrier(CarrierInput):
    """Procedurally generated dragon curve pattern."""

    def __init__(self, size: int = 64, iterations: int = 10, line_width: float = 2.0):
        super().__init__(size)
        self.iterations = iterations
        self.line_width = line_width

    def _generate_base(self) -> np.ndarray:
        from .patterns import generate_dragon_curve
        return generate_dragon_curve(self.size, self.iterations, self.line_width)


class GradientCarrier(CarrierInput):
    """Linear or radial gradient pattern."""

    def __init__(self, size: int = 64, gradient_type: str = 'linear', angle: float = 0.0):
        super().__init__(size)
        self.gradient_type = gradient_type
        self.angle = angle

    def _generate_base(self) -> np.ndarray:
        from .patterns import generate_gradient
        return generate_gradient(self.size, self.gradient_type)


class ArrayCarrier(CarrierInput):
    """Carrier from existing numpy array."""

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

    def _generate_base(self) -> np.ndarray:
        # Resize if not square
        if self._array.shape[0] != self._array.shape[1]:
            from scipy.ndimage import zoom
            target = max(self._array.shape)
            factors = (target / self._array.shape[0], target / self._array.shape[1])
            return zoom(self._array, factors, order=1)
        return self._array.copy()
