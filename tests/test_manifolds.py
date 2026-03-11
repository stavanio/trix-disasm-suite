"""
Unit tests for TRiX manifold projections.

Verifies:
1. Projection output lies on manifold
2. Projection is minimal (closest point)
3. Safe inputs pass through unchanged
4. Deterministic with fixed seed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from trix.manifolds import (helical_projection, thermal_projection,
                            planar_projection, tangential_projection)
from trix.governor import TRiXGovernor


def test_helical_manifold_membership():
    """Projected point must satisfy v_z = k * omega_z."""
    pitches = [0.5, 1.0, 1.25, 1.5, 2.0]
    for pitch in pitches:
        k = pitch / (2 * np.pi)
        for _ in range(1000):
            v_z = np.random.uniform(-10, 10)
            omega_z = np.random.uniform(-10, 10)
            v_safe, omega_safe = helical_projection(v_z, omega_z, pitch)
            # Check manifold membership
            assert abs(v_safe - k * omega_safe) < 1e-10, \
                f"Manifold violation: v={v_safe}, k*w={k*omega_safe}"
    print("  PASS: helical_manifold_membership")


def test_helical_minimality():
    """Projection must be the closest point on manifold."""
    pitch = 1.25
    k = pitch / (2 * np.pi)
    for _ in range(1000):
        v_z = np.random.uniform(-10, 10)
        omega_z = np.random.uniform(-10, 10)
        v_safe, omega_safe = helical_projection(v_z, omega_z, pitch)

        # Distance to projected point
        d_proj = np.sqrt((v_z - v_safe)**2 + (omega_z - omega_safe)**2)

        # Check against random manifold points
        for _ in range(100):
            w = np.random.uniform(-20, 20)
            v = k * w
            d_random = np.sqrt((v_z - v)**2 + (omega_z - w)**2)
            assert d_proj <= d_random + 1e-10, "Projection is not minimal"
    print("  PASS: helical_minimality")


def test_helical_safe_passthrough():
    """Points already on manifold should pass through unchanged."""
    pitch = 1.25
    k = pitch / (2 * np.pi)
    for _ in range(100):
        omega = np.random.uniform(-10, 10)
        v = k * omega  # already on manifold
        v_safe, omega_safe = helical_projection(v, omega, pitch)
        assert abs(v - v_safe) < 1e-10
        assert abs(omega - omega_safe) < 1e-10
    print("  PASS: helical_safe_passthrough")


def test_thermal_projection():
    """Power constraint: F*v <= P_max."""
    for _ in range(1000):
        F = np.random.uniform(1, 50)
        v = np.random.uniform(0, 1)
        P_max = np.random.uniform(1, 20)
        F_safe, v_safe = thermal_projection(F, v, P_max)
        assert F_safe * abs(v_safe) <= P_max + 1e-10, \
            f"Thermal violation: P={F_safe*abs(v_safe)}, P_max={P_max}"
    print("  PASS: thermal_projection")


def test_thermal_passthrough():
    """Safe actions should pass through unchanged."""
    F, v, P_max = 5.0, 0.1, 10.0  # power = 0.5 < 10
    F_safe, v_safe = thermal_projection(F, v, P_max)
    assert abs(F - F_safe) < 1e-10
    assert abs(v - v_safe) < 1e-10
    print("  PASS: thermal_passthrough")


def test_planar_projection():
    """Tilt torques must be zeroed."""
    for _ in range(1000):
        wrench = np.random.uniform(-10, 10, 6)
        result = planar_projection(wrench, F_frac=30.0)
        assert abs(result[3]) < 1e-10, "tau_x not zeroed"
        assert abs(result[4]) < 1e-10, "tau_y not zeroed"
        assert result[2] <= 30.0 + 1e-10, "F_z not clipped"
    print("  PASS: planar_projection")


def test_governor_screw():
    """TRiX governor SCREW task integration test."""
    gov = TRiXGovernor('SCREW')
    for _ in range(1000):
        action = np.random.uniform(-1, 1, 3).astype(np.float32)
        safe = gov.project(action)
        # Radial must be zero
        assert abs(safe[2]) < 1e-10, f"Radial not zeroed: {safe[2]}"
        # Torque within bounds
        assert abs(safe[0]) <= 0.55 + 1e-5
    print("  PASS: governor_screw")


def test_governor_deterministic():
    """Same input -> same output."""
    gov = TRiXGovernor('SCREW')
    action = np.array([0.3, 0.7, -0.5], dtype=np.float32)
    r1 = gov.project(action)
    r2 = gov.project(action)
    assert np.allclose(r1, r2), "Non-deterministic projection"
    print("  PASS: governor_deterministic")


def run_all_tests():
    print("=" * 60)
    print("TRiX Manifold Projection Tests")
    print("=" * 60)

    test_helical_manifold_membership()
    test_helical_minimality()
    test_helical_safe_passthrough()
    test_thermal_projection()
    test_thermal_passthrough()
    test_planar_projection()
    test_governor_screw()
    test_governor_deterministic()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
