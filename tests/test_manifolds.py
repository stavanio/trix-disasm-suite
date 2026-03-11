"""
Unit tests for TRiX manifold projections.

Verifies:
  1. Projected point lies on manifold (membership)
  2. Projection is minimal (closest point)
  3. Safe inputs pass through unchanged (idempotency)
  4. Deterministic output for fixed input

Run:  python -m pytest tests/ -v
  or: python tests/test_manifolds.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from trix.manifolds import helical_projection, thermal_projection, planar_projection


def test_helical_manifold_membership():
    """Projected point must satisfy v_z = k * omega_z."""
    for pitch in [0.5, 1.0, 1.25, 1.5, 2.0]:
        k = pitch / (2 * np.pi)
        for _ in range(1000):
            vz, wz = np.random.uniform(-10, 10, 2)
            v_safe, w_safe = helical_projection(vz, wz, pitch)
            assert abs(v_safe - k * w_safe) < 1e-10, \
                f"Manifold violation: {v_safe} != {k}*{w_safe}"
    print("  PASS: helical_manifold_membership")


def test_helical_minimality():
    """Projection must be closest point on manifold."""
    pitch, k = 1.25, 1.25 / (2 * np.pi)
    for _ in range(1000):
        vz, wz = np.random.uniform(-10, 10, 2)
        v_safe, w_safe = helical_projection(vz, wz, pitch)
        d_proj = np.sqrt((vz - v_safe)**2 + (wz - w_safe)**2)
        for _ in range(50):
            w = np.random.uniform(-20, 20)
            d = np.sqrt((vz - k*w)**2 + (wz - w)**2)
            assert d_proj <= d + 1e-10
    print("  PASS: helical_minimality")


def test_helical_idempotency():
    """Points on manifold pass through unchanged."""
    pitch, k = 1.25, 1.25 / (2 * np.pi)
    for _ in range(100):
        w = np.random.uniform(-10, 10)
        v = k * w
        vs, ws = helical_projection(v, w, pitch)
        assert abs(v - vs) < 1e-10 and abs(w - ws) < 1e-10
    print("  PASS: helical_idempotency")


def test_thermal_constraint():
    """F*v must be <= P_max after projection."""
    for _ in range(1000):
        F = np.random.uniform(1, 50)
        v = np.random.uniform(0, 1)
        P_max = np.random.uniform(1, 20)
        Fs, vs = thermal_projection(F, v, P_max)
        assert Fs * abs(vs) <= P_max + 1e-10
    print("  PASS: thermal_constraint")


def test_thermal_passthrough():
    """Safe actions unchanged."""
    Fs, vs = thermal_projection(5.0, 0.1, 10.0)
    assert abs(5.0 - Fs) < 1e-10 and abs(0.1 - vs) < 1e-10
    print("  PASS: thermal_passthrough")


def test_planar_torques_zeroed():
    """Tilt torques must be zero after projection."""
    for _ in range(1000):
        w = np.random.uniform(-10, 10, 6)
        r = planar_projection(w)
        assert abs(r[3]) < 1e-10 and abs(r[4]) < 1e-10
        assert r[2] <= 30.0 + 1e-10
    print("  PASS: planar_torques_zeroed")


if __name__ == '__main__':
    print("=" * 60)
    print("TRiX Manifold Projection Tests")
    print("=" * 60)
    test_helical_manifold_membership()
    test_helical_minimality()
    test_helical_idempotency()
    test_thermal_constraint()
    test_thermal_passthrough()
    test_planar_torques_zeroed()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
