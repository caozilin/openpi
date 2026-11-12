"""Explicit LIBERO state8/action7 + six signed tolerance labels contract."""
import numpy as np

SOURCE_SCHEMA = 'libero_joint13/1.0'
TOLERANCE_ORDER = ('rx_negative', 'rx_positive', 'ry_negative', 'ry_positive', 'rz_negative', 'rz_positive')

def validate_info(info):
    if info.get('source_schema') != SOURCE_SCHEMA or info.get('fps') != 20:
        raise ValueError('Require audited LIBERO joint13 schema at 20Hz')
    expected = {'state': ('float32', [8]), 'actions': ('float32', [7]),
                'rotation_tolerance': ('float32', [6]), 'trajectory_is_tolerance': ('int64', [1])}
    for name, (dtype, shape) in expected.items():
        feature = info.get('features', {}).get(name, {})
        if feature.get('dtype') != dtype or feature.get('shape') != shape:
            raise ValueError(f'Invalid LIBERO joint13 feature {name}: {feature}')
    contract = info.get('joint13_contract', {})
    if contract.get('rotation_tolerance_units') != 'rad' or contract.get('rotation_tolerance_order') != list(TOLERANCE_ORDER):
        raise ValueError('Require radians and explicit signed directional order')
    if contract.get('trajectory_types') != {'0': 'nominal', '1': 'tolerance'}:
        raise ValueError('Require explicit trajectory type meanings')

def validate_tolerance(values, *, leading_shape=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 6 or (leading_shape is not None and values.shape[:-1] != leading_shape):
        raise ValueError(f'Invalid rotation_tolerance shape: {values.shape}')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > np.float32(np.pi / 6) + 1e-7):
        raise ValueError('Tolerance labels must be finite radians in [0, pi/6]; missing labels cannot be zero-filled')
    return values
