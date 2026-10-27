import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.libero_tasktol_compat.annotate_original_closing_vertical import closing_vertical_angle_deg


def test_unoriented_angle_and_near_vertical_resolution():
    theta = np.radians([0., 1e-8, 1., 45., 90.])
    axes = np.column_stack([np.sin(theta), np.zeros(len(theta)), np.cos(theta)])
    np.testing.assert_allclose(closing_vertical_angle_deg(axes), np.degrees(theta), atol=1e-12)
    np.testing.assert_allclose(closing_vertical_angle_deg(-axes), np.degrees(theta), atol=1e-12)


def test_site_y_is_not_the_physical_closing_axis_after_libero_mount_rotation():
    hand = Rotation.from_euler('x', 70, degrees=True).as_matrix()
    mount = Rotation.from_euler('z', -90, degrees=True).as_matrix()
    finger = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    physical = hand @ mount @ finger @ np.array([0., 1., 0.])
    np.testing.assert_allclose(closing_vertical_angle_deg(physical), 20., atol=1e-12)
    np.testing.assert_allclose(closing_vertical_angle_deg((hand @ mount)[:, 1]), 90., atol=1e-12)


def test_invalid_axis_is_rejected():
    with pytest.raises(ValueError):
        closing_vertical_angle_deg([0., 0., 0.])
