"""
Unified Texture Synthesis Module

THE single source of truth for texture synthesis in the itten project.
All texture operations should use this module.

Primary API:
    synthesize()      - THE function for texture synthesis
    TextureResult     - Dataclass returned by synthesize()

Pattern Generators:
    generate_amongus()
    generate_checkerboard()
    generate_noise()
    generate_dragon_curve()

Normal Maps:
    height_to_normals()

Example:
    >>> from texture import synthesize
    >>> result = synthesize('amongus', 'checkerboard', theta=0.5)
    >>> result.height_field.shape
    (64, 64)
    >>> result.normal_map.shape
    (64, 64, 3)

For CLI usage:
    python -m texture --demo
    python -m texture --carrier amongus --operand noise
    python -m texture --help

For headless batch processing:
    python -m texture.interfaces.headless config.json
"""

__version__ = "1.0.0"

# Core API
from .core import (
    synthesize,
    TextureResult,
    quick_synthesize,
    normalize_to_01,
)

# Pattern generators
from .patterns import (
    generate_amongus,
    generate_checkerboard,
    generate_noise,
    generate_dragon_curve,
    generate_gradient,
    generate_tiled_amongus,
    generate_varied_amongus,
)

# Normal map utilities
from .normals import (
    height_to_normals,
    normals_to_image,
    combine_height_fields,
    visualize_normals_lit,
)

# Carrier/Operand classes
from .carriers import (
    CarrierInput,
    AmongusCarrier,
    CheckerboardCarrier,
    NoiseCarrier,
    DragonCurveCarrier,
    GradientCarrier,
    ArrayCarrier,
)

from .operands import (
    OperandInput,
    CheckerboardOperand,
    NoiseOperand,
    SolidOperand,
    GradientOperand,
    CircleOperand,
    ArrayOperand,
)

# Contour extraction
from .contours import (
    extract_contours,
    extract_nodal_lines,
    extract_partition_boundary,
    carrier_edge_field,
)

# I/O utilities
from .io import (
    export_heightmap_png,
    export_normalmap_png,
    export_obj_with_uv,
    export_all,
    load_image_as_array,
    load_config,
    save_config,
)

__all__ = [
    # Core API
    'synthesize',
    'TextureResult',
    'quick_synthesize',
    'normalize_to_01',

    # Pattern generators
    'generate_amongus',
    'generate_checkerboard',
    'generate_noise',
    'generate_dragon_curve',
    'generate_gradient',
    'generate_tiled_amongus',
    'generate_varied_amongus',

    # Normal maps
    'height_to_normals',
    'normals_to_image',
    'combine_height_fields',
    'visualize_normals_lit',

    # Carrier classes
    'CarrierInput',
    'AmongusCarrier',
    'CheckerboardCarrier',
    'NoiseCarrier',
    'DragonCurveCarrier',
    'GradientCarrier',
    'ArrayCarrier',

    # Operand classes
    'OperandInput',
    'CheckerboardOperand',
    'NoiseOperand',
    'SolidOperand',
    'GradientOperand',
    'CircleOperand',
    'ArrayOperand',

    # Contour extraction
    'extract_contours',
    'extract_nodal_lines',
    'extract_partition_boundary',
    'carrier_edge_field',

    # I/O
    'export_heightmap_png',
    'export_normalmap_png',
    'export_obj_with_uv',
    'export_all',
    'load_image_as_array',
    'load_config',
    'save_config',
]
