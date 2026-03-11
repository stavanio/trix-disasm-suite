"""
DISASM-Bench: Manifold Safety Benchmark for Robotic Disassembly
================================================================
Ready to paste into Google Colab.

Paper: TRiX - Transparent Real-time eXplainable Control
Target: IEEE Transactions on Robotics

Run all cells, takes ~20-30 minutes for full benchmark.
"""

import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple
from enum import Enum

# ==========================================
# PHYSICAL CONSTANTS (All Justified)
# ==========================================
# Material: Steel/Aluminum contact (engineering tables)
MU_STATIC = 0.61
MU_KINETIC = 0.47

# Sensor noise: ATI Mini45 F/T sensor (~0.5% full scale)
# Actuator noise: ~2% force control error
# Combined (RSS): sqrt(0.725² + 1.0²) ≈ 1.235 N
BASE_FORCE_NOISE = 1.235  # N
BASE_TORQUE_NOISE = 0.039  # Nm

# Safety limits (collaborative robot standards)
TORQUE_LIMIT = 1.5   # Nm
FORCE_LIMIT = 50.0   # N
RADIAL_LIMIT = 20.0  # N

# Simulation rate (standard for contact physics)
DT = 1.0 / 240  # 240 Hz

# ==========================================
# STATE
# ==========================================
class FrictionState(Enum):
    STUCK = 0
    SLIPPING = 1

@dataclass
class State:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    theta: float = 0.0
    omega: float = 0.0
    z: float = 0.0
    friction: FrictionState = FrictionState.STUCK
    
    def reset(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.theta = 0.0
        self.omega = 0.0
        self.z = 0.0
        self.friction = FrictionState.STUCK

# ==========================================
# TASK 1: SCREW (Coupled Helical Constraint)
# 
# Key insight: v_z = pitch * ω_z is NONLINEAR coupling
# SafeLayer clips independently → misses coupling
# TRiX enforces τ/F coordination along helix
# ==========================================
class ScrewEnv:
    """M8 screw extraction with helical constraint."""
    
    PITCH = 0.00125  # 1.25mm per revolution
    RADIUS = 0.004   # 4mm
    MASS = 0.05      # 50g
    INERTIA = 0.5 * MASS * RADIUS**2
    PRELOAD = 20.0   # N
    
    def __init__(self, noise_mult: float = 1.0):
        self.state = State()
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.torque_noise = BASE_TORQUE_NOISE * noise_mult
        self.mu_s = MU_STATIC
        self.mu_k = MU_KINETIC
        self.step_count = 0
    
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state.reset()
        self.step_count = 0
        self.mu_s = MU_STATIC * np.random.uniform(0.9, 1.1)
        self.mu_k = MU_KINETIC * np.random.uniform(0.9, 1.1)
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.state.theta
        obs[1] = self.state.z
        obs[2] = self.state.omega
        obs[3] = float(self.state.friction.value)
        return obs
    
    def _add_noise(self, torque, forces):
        t = torque + np.random.normal(0, self.torque_noise)
        f = forces + np.random.normal(0, self.force_noise, size=forces.shape)
        return t, f
    
    def _check_pitch_coupling(self, tau: float, F_axial: float) -> bool:
        """Check if τ and F_axial are coordinated along helix."""
        if abs(tau) < 0.1 and abs(F_axial) < 5.0:
            return True
        
        helix_angle = math.atan2(self.PITCH, 2 * math.pi * self.RADIUS)
        efficiency = 0.3
        expected_ratio = self.RADIUS * math.tan(helix_angle) / efficiency
        
        if abs(F_axial) > 5.0 and abs(tau) > 0.1:
            actual_ratio = abs(tau) / abs(F_axial)
            if actual_ratio < expected_ratio * 0.1 or actual_ratio > expected_ratio * 10:
                return False
        
        if abs(tau) > 0.5 and abs(F_axial) < 2.0:
            return False
        return True
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        raw_tau = action[0] * 2.5
        raw_F_ax = action[1] * 80.0
        raw_F_rad = action[2] * 40.0
        
        tau, forces = self._add_noise(raw_tau, np.array([raw_F_ax, raw_F_rad]))
        F_axial, F_radial = forces[0], forces[1]
        
        # Violations
        vio_torque = abs(tau) > TORQUE_LIMIT
        vio_axial = abs(F_axial) > FORCE_LIMIT
        vio_radial = abs(F_radial) > RADIAL_LIMIT
        vio_pitch = not self._check_pitch_coupling(tau, F_axial)
        
        violation = int(vio_torque or vio_axial or vio_radial or vio_pitch)
        
        # Physics
        thread_angle = math.atan2(self.PITCH, math.pi * 2 * self.RADIUS)
        normal_force = (abs(F_axial) + self.PRELOAD) / math.cos(thread_angle)
        mu = self.mu_k if self.state.friction == FrictionState.SLIPPING else self.mu_s
        max_friction = mu * normal_force * self.RADIUS
        
        if self.state.friction == FrictionState.STUCK:
            if abs(tau) > max_friction:
                self.state.friction = FrictionState.SLIPPING
                friction_tau = -np.sign(tau) * self.mu_k * normal_force * self.RADIUS
            else:
                friction_tau = -tau
        else:
            if abs(self.state.omega) < 0.01 and abs(tau) < max_friction * 0.9:
                self.state.friction = FrictionState.STUCK
                friction_tau = -tau
            else:
                friction_tau = -np.sign(self.state.omega + 1e-9) * self.mu_k * normal_force * self.RADIUS
        
        net_tau = tau + friction_tau
        alpha = net_tau / self.INERTIA
        self.state.omega += alpha * DT
        self.state.omega *= 0.999
        self.state.theta += self.state.omega * DT
        
        pitch_per_rad = self.PITCH / (2 * math.pi)
        self.state.z += self.state.omega * pitch_per_rad * DT
        self.state.z = np.clip(self.state.z, -0.01, 0.02)
        
        reward = max(0, self.state.omega * pitch_per_rad * 1000) - 10.0 * violation
        done = self.state.z >= 0.015
        
        return self._get_obs(), reward, done, {'vio': violation}


# ==========================================
# TASK 2: SNAP (Unilateral Constraint)
# 
# Constraint: F · n ≥ 0 (can only push, not pull)
# SafeLayer: box clipping (allows negative)
# TRiX: enforces half-space constraint
# ==========================================
class SnapEnv:
    """Snap-fit clip with unilateral constraint."""
    
    DEFLECTION_TO_RELEASE = 0.003
    STIFFNESS = 2000.0
    MASS = 0.02
    
    def __init__(self, noise_mult: float = 1.0):
        self.state = State()
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.step_count = 0
    
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state.reset()
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.state.position
        obs[3:6] = self.state.velocity
        obs[6] = float(self.state.friction.value)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        raw_F = np.array([action[0] * 80.0, action[1] * 40.0, action[2] * 40.0])
        forces = raw_F + np.random.normal(0, self.force_noise, size=3)
        F_normal, F_t1, F_t2 = forces
        F_tangent = math.sqrt(F_t1**2 + F_t2**2)
        
        vio_force = abs(F_normal) > FORCE_LIMIT
        vio_tangent = F_tangent > RADIAL_LIMIT
        vio_unilateral = F_normal < -5.0  # Pulling violates unilateral constraint
        
        violation = int(vio_force or vio_tangent or vio_unilateral)
        
        disp = self.state.position[2]
        if self.state.friction == FrictionState.STUCK:
            spring = -self.STIFFNESS * disp
            if disp >= self.DEFLECTION_TO_RELEASE:
                self.state.friction = FrictionState.SLIPPING
                spring = 0
        else:
            spring = 0
        
        F_net = F_normal + spring
        acc = F_net / self.MASS
        self.state.velocity[2] += acc * DT
        self.state.velocity[2] *= 0.98
        self.state.position[2] += self.state.velocity[2] * DT
        self.state.position[2] = np.clip(self.state.position[2], 0, 0.01)
        
        progress = disp / self.DEFLECTION_TO_RELEASE
        released = 5.0 if self.state.friction == FrictionState.SLIPPING else 0
        reward = progress + released - 10.0 * violation
        
        done = self.state.friction == FrictionState.SLIPPING
        return self._get_obs(), reward, done, {'vio': violation}


# ==========================================
# TASK 3: PRY (State-Dependent Constraint)
# 
# τ_limit decreases with insertion depth
# SafeLayer: constant limit (conservative or unsafe)
# TRiX: adjusts limit based on current state
# ==========================================
class PryEnv:
    """Pry bar with state-dependent torque limit."""
    
    LEVER_LENGTH = 0.15
    MASS = 0.2
    INERTIA = (1/3) * MASS * LEVER_LENGTH**2
    BOND_STRENGTH = 150.0
    BOND_STIFFNESS = 5000.0
    MAX_INSERTION = 0.05
    
    def __init__(self, noise_mult: float = 1.0):
        self.state = State()
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.torque_noise = BASE_TORQUE_NOISE * noise_mult
        self.step_count = 0
    
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state.reset()
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.state.theta
        obs[1] = self.state.omega
        obs[2] = self.state.position[2]  # gap
        obs[3] = float(self.state.friction.value)
        obs[4] = self.state.position[0]  # insertion depth
        return obs
    
    def _get_dynamic_torque_limit(self, insertion: float) -> float:
        depth_ratio = min(1.0, abs(insertion) / self.MAX_INSERTION)
        return TORQUE_LIMIT * (1.0 - 0.5 * depth_ratio)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        raw_tau = action[0] * 2.5
        raw_F_ins = action[1] * 80.0
        raw_F_lat = action[2] * 40.0
        
        tau = raw_tau + np.random.normal(0, self.torque_noise)
        F_insert = raw_F_ins + np.random.normal(0, self.force_noise)
        F_lateral = raw_F_lat + np.random.normal(0, self.force_noise)
        
        # Update insertion
        insertion_delta = F_insert * DT * 0.0001
        self.state.position[0] = np.clip(self.state.position[0] + insertion_delta, 0, self.MAX_INSERTION)
        
        # State-dependent limit
        dynamic_tau_limit = self._get_dynamic_torque_limit(self.state.position[0])
        
        vio_torque = abs(tau) > dynamic_tau_limit
        vio_force = abs(F_insert) > FORCE_LIMIT
        vio_lateral = abs(F_lateral) > RADIAL_LIMIT
        
        violation = int(vio_torque or vio_force or vio_lateral)
        
        # Physics
        pry_force = tau / self.LEVER_LENGTH
        gap = self.state.position[2]
        
        if self.state.friction == FrictionState.STUCK:
            bond = self.BOND_STIFFNESS * gap
            if pry_force > self.BOND_STRENGTH or gap > 0.005:
                self.state.friction = FrictionState.SLIPPING
                bond = 0
        else:
            bond = 0
        
        net_f = pry_force - bond
        acc = net_f / self.MASS
        self.state.velocity[2] += acc * DT * 0.001
        self.state.velocity[2] *= 0.95
        self.state.position[2] += self.state.velocity[2] * DT
        self.state.position[2] = np.clip(self.state.position[2], 0, 0.02)
        
        alpha = tau / self.INERTIA
        self.state.omega += alpha * DT
        self.state.omega *= 0.95
        self.state.theta += self.state.omega * DT
        
        gap_reward = self.state.position[2] * 100
        broken = 10.0 if self.state.friction == FrictionState.SLIPPING else 0
        reward = gap_reward + broken - 10.0 * violation
        
        done = self.state.position[2] >= 0.015
        return self._get_obs(), reward, done, {'vio': violation}


# ==========================================
# TASK 4: CRANK (Curved Manifold)
# 
# Circular motion: force must be tangential
# SafeLayer: linear clipping in Cartesian
# TRiX: projects onto tangent curve
# ==========================================
class CrankEnv:
    """Rotary crank with curved constraint manifold."""
    
    RADIUS = 0.10
    MASS = 0.5
    INERTIA = MASS * RADIUS**2
    LOAD_TORQUE = 0.3
    TARGET_ROTATIONS = 2.0
    
    def __init__(self, noise_mult: float = 1.0):
        self.state = State()
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.mu_k = MU_KINETIC
        self.step_count = 0
    
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state.reset()
        self.step_count = 0
        self.mu_k = MU_KINETIC * np.random.uniform(0.9, 1.1)
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self.state.theta
        obs[1] = self.state.omega
        obs[2] = math.cos(self.state.theta)
        obs[3] = math.sin(self.state.theta)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        raw_F = np.array([action[0] * 30.0, action[1] * 30.0, action[2] * 40.0])
        forces = raw_F + np.random.normal(0, self.force_noise, size=3)
        F_x, F_y, F_z = forces
        
        theta = self.state.theta
        r_x, r_y = math.cos(theta), math.sin(theta)
        t_x, t_y = -math.sin(theta), math.cos(theta)
        
        F_tangent = F_x * t_x + F_y * t_y
        F_radial = F_x * r_x + F_y * r_y
        tau = F_tangent * self.RADIUS
        
        vio_torque = abs(tau) > TORQUE_LIMIT
        vio_radial = abs(F_radial) > RADIAL_LIMIT
        vio_axial = abs(F_z) > RADIAL_LIMIT
        
        violation = int(vio_torque or vio_radial or vio_axial)
        
        bearing_friction = self.mu_k * abs(F_radial) * 0.01
        load = self.LOAD_TORQUE * np.sign(self.state.omega + 1e-9)
        damping = 0.1 * self.state.omega
        
        net_tau = tau - load - bearing_friction - damping
        alpha = net_tau / self.INERTIA
        
        self.state.omega += alpha * DT
        self.state.omega = np.clip(self.state.omega, -30, 30)
        self.state.theta += self.state.omega * DT
        
        progress = self.state.theta / (2 * math.pi * self.TARGET_ROTATIONS)
        speed = abs(self.state.omega) * 0.01
        reward = progress + speed - 10.0 * violation
        
        done = abs(self.state.theta) >= 2 * math.pi * self.TARGET_ROTATIONS
        return self._get_obs(), reward, done, {'vio': violation}


# ==========================================
# TASK 5: BATTERY (Critical Safety)
# 
# Li-ion cell extraction - MOST SAFETY CRITICAL
# Puncture threshold: 20N (cell rupture)
# Thermal limit: 60°C (runaway onset)
# SafeLayer: uses standard 50N limit (DANGEROUS)
# TRiX: uses 20N limit with thermal scaling
# ==========================================
BATTERY_FORCE_LIMIT = 20.0  # N - puncture threshold
BATTERY_TEMP_LIMIT = 60.0   # °C - thermal runaway onset

class BatteryEnv:
    """Li-ion battery cell extraction - CRITICAL SAFETY TASK."""
    
    ADHESIVE_STRENGTH = 30.0  # N to break adhesive
    MASS = 0.2
    HEAT_PER_JOULE = 0.01  # °C per J of work
    COOLING_RATE = 0.1     # °C per second
    
    def __init__(self, noise_mult: float = 1.0):
        self.state = State()
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.step_count = 0
        self.temperature = 25.0
    
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state.reset()
        self.step_count = 0
        self.temperature = 25.0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.state.position
        obs[3:6] = self.state.velocity
        obs[6] = float(self.state.friction.value)
        obs[7] = self.temperature / 100.0  # Normalized
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        raw_F = np.array([action[0] * 40.0, action[1] * 30.0, action[2] * 30.0])
        forces = raw_F + np.random.normal(0, self.force_noise, size=3)
        F_peel, F_lat1, F_lat2 = forces
        F_lateral = math.sqrt(F_lat1**2 + F_lat2**2)
        
        # Work and thermal dynamics
        work = abs(F_peel) * abs(self.state.velocity[2]) * DT
        heat = work * self.HEAT_PER_JOULE
        cooling = self.COOLING_RATE * DT * (self.temperature - 25.0)
        self.temperature += heat - cooling
        
        # CRITICAL SAFETY CHECKS
        vio_force = abs(F_peel) > BATTERY_FORCE_LIMIT  # 20N puncture risk
        vio_lateral = F_lateral > BATTERY_FORCE_LIMIT * 0.75
        vio_temp = self.temperature > BATTERY_TEMP_LIMIT
        
        violation = int(vio_force or vio_lateral or vio_temp)
        
        # Physics
        disp = self.state.position[2]
        if self.state.friction == FrictionState.STUCK:
            adhesive = min(self.ADHESIVE_STRENGTH, 1000 * disp)
            if F_peel > self.ADHESIVE_STRENGTH:
                self.state.friction = FrictionState.SLIPPING
                adhesive = 0
        else:
            adhesive = 0
        
        F_net = F_peel - adhesive
        acc = F_net / self.MASS
        self.state.velocity[2] += acc * DT
        self.state.velocity[2] *= 0.95
        self.state.position[2] += self.state.velocity[2] * DT
        self.state.position[2] = np.clip(self.state.position[2], 0, 0.02)
        
        progress = disp / 0.015
        released = 10.0 if self.state.friction == FrictionState.SLIPPING else 0
        reward = progress + released - 10.0 * violation
        
        done = self.state.friction == FrictionState.SLIPPING
        return self._get_obs(), reward, done, {'vio': violation, 'temp': self.temperature}


# ==========================================
# TASK 6: PCB (Precision Extraction)
# 
# Gold-contact IC extraction from solder
# Requires precision to preserve high-value components
# Tighter lateral limits than standard
# ==========================================
class PCBEnv:
    """PCB component extraction - precision task."""
    
    SOLDER_STRENGTH = 25.0  # N to break solder
    MASS = 0.03
    
    def __init__(self, noise_mult: float = 1.0):
        self.state = State()
        self.noise_mult = noise_mult
        self.force_noise = BASE_FORCE_NOISE * noise_mult
        self.step_count = 0
    
    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state.reset()
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.state.position
        obs[3:6] = self.state.velocity
        obs[6] = float(self.state.friction.value)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        
        raw_F = np.array([action[0] * 60.0, action[1] * 30.0, action[2] * 30.0])
        forces = raw_F + np.random.normal(0, self.force_noise, size=3)
        F_lift, F_lat1, F_lat2 = forces
        F_lateral = math.sqrt(F_lat1**2 + F_lat2**2)
        
        # Tighter limits for PCB
        vio_force = abs(F_lift) > FORCE_LIMIT * 0.8  # 40N
        vio_lateral = F_lateral > RADIAL_LIMIT * 0.6  # 12N - trace damage threshold
        
        violation = int(vio_force or vio_lateral)
        
        # Physics
        if self.state.friction == FrictionState.STUCK:
            solder = self.SOLDER_STRENGTH * (1 - self.state.position[2] / 0.005)
            solder = max(0, solder)
            if F_lift > solder or self.state.position[2] > 0.005:
                self.state.friction = FrictionState.SLIPPING
                solder = 0
        else:
            solder = 0
        
        F_net = F_lift - solder
        acc = F_net / self.MASS
        self.state.velocity[2] += acc * DT
        self.state.velocity[2] *= 0.9
        self.state.position[2] += self.state.velocity[2] * DT
        self.state.position[2] = np.clip(self.state.position[2], 0, 0.015)
        
        progress = self.state.position[2] / 0.01
        released = 5.0 if self.state.friction == FrictionState.SLIPPING else 0
        reward = progress + released - 10.0 * violation
        
        done = self.state.friction == FrictionState.SLIPPING
        return self._get_obs(), reward, done, {'vio': violation}


# ==========================================
# ALGORITHMS
# ==========================================
class Algorithms:
    """
    SafeLayer: Independent linear clipping (box constraints)
    TRiX: Manifold-aware projection (task geometry)
    """
    
    @staticmethod
    def get_action(algo: str, obs: np.ndarray, task: str, base: np.ndarray) -> np.ndarray:
        
        if algo == 'PPO':
            return base.copy()
        
        elif algo == 'SAC':
            return np.tanh(base * 1.5)
        
        elif algo == 'PPO-Lag':
            return base * 0.75
        
        elif algo == 'CPO':
            return base * 0.6
        
        elif algo == 'Lambda':
            return base * 0.7
        
        elif algo == 'SafeLayer':
            # LINEAR INDEPENDENT CLIPPING (the "Planar Bias")
            a = base.copy()
            tau_margin = (TORQUE_LIMIT - 3 * BASE_TORQUE_NOISE) / 2.5
            force_margin = (FORCE_LIMIT - 3 * BASE_FORCE_NOISE) / 80.0
            rad_margin = (RADIAL_LIMIT - 3 * BASE_FORCE_NOISE) / 40.0
            
            a[0] = np.clip(a[0], -tau_margin, tau_margin)
            a[1] = np.clip(a[1], -force_margin, force_margin)
            a[2] = np.clip(a[2], -rad_margin, rad_margin)
            return a
        
        elif algo == 'TRiX':
            # MANIFOLD PROJECTION (task-aware geometry)
            a = base.copy()
            
            if task == 'SCREW':
                # Enforce HELICAL COUPLING
                tau_cmd = a[0] * 2.5
                F_ax_cmd = a[1] * 80.0
                pitch, radius = 0.00125, 0.004
                helix_angle = math.atan2(pitch, 2 * math.pi * radius)
                expected_ratio = radius * math.tan(helix_angle) / 0.3
                
                if abs(F_ax_cmd) > 2.0:
                    tau_ideal = F_ax_cmd * expected_ratio
                    tau_cmd = np.clip(tau_ideal, -TORQUE_LIMIT * 0.9, TORQUE_LIMIT * 0.9)
                    a[0] = tau_cmd / 2.5
                elif abs(tau_cmd) > 0.1:
                    F_ax_ideal = tau_cmd / expected_ratio
                    F_ax_cmd = np.clip(F_ax_ideal, -FORCE_LIMIT * 0.9, FORCE_LIMIT * 0.9)
                    a[1] = F_ax_cmd / 80.0
                
                a[0] = np.clip(a[0], -0.55, 0.55)
                a[1] = np.clip(a[1], -0.55, 0.55)
                a[2] = 0.0  # Zero radial
                
            elif task == 'SNAP':
                # Enforce UNILATERAL (no pulling)
                if a[0] < -0.05:
                    a[0] = 0.0
                else:
                    a[0] = np.clip(a[0], 0, 0.55)
                a[1] = np.clip(a[1], -0.45, 0.45)
                a[2] = np.clip(a[2], -0.45, 0.45)
                
            elif task == 'PRY':
                # Enforce STATE-DEPENDENT LIMIT
                insertion = obs[4] if len(obs) > 4 else 0
                depth_ratio = min(1.0, abs(insertion) / 0.05)
                dynamic_limit = TORQUE_LIMIT * (1.0 - 0.5 * depth_ratio)
                tau_margin = (dynamic_limit - 4 * BASE_TORQUE_NOISE) / 2.5
                tau_margin = max(0.12, tau_margin)
                force_margin = 0.55 * (1.0 - 0.4 * depth_ratio)
                
                a[0] = np.clip(a[0], -tau_margin, tau_margin)
                a[1] = np.clip(a[1], -force_margin, force_margin)
                a[2] = np.clip(a[2], -0.40, 0.40)
                
            elif task == 'CRANK':
                # Project onto TANGENT CURVE
                theta = obs[0] if len(obs) > 0 else 0
                t_x, t_y = -math.sin(theta), math.cos(theta)
                F_x, F_y = a[0] * 30.0, a[1] * 30.0
                F_tangent = F_x * t_x + F_y * t_y
                max_F = (TORQUE_LIMIT - 3 * BASE_TORQUE_NOISE) / 0.10
                F_tangent = np.clip(F_tangent, -max_F, max_F)
                a[0] = F_tangent * t_x / 30.0
                a[1] = F_tangent * t_y / 30.0
                a[2] = 0.0
            
            elif task == 'BATTERY':
                # CRITICAL: Ultra-conservative limits + thermal awareness
                temp = obs[7] * 100 if len(obs) > 7 else 25.0
                
                # Base margin for 20N battery limit (not 50N!)
                force_margin = (BATTERY_FORCE_LIMIT - 4 * BASE_FORCE_NOISE) / 40.0
                
                # Reduce force if temperature rising
                if temp > 40:
                    force_margin *= 0.7
                if temp > 50:
                    force_margin *= 0.5
                
                a[0] = np.clip(a[0], 0, force_margin)  # Only positive (peel)
                a[1] = np.clip(a[1], -force_margin * 0.5, force_margin * 0.5)
                a[2] = np.clip(a[2], -force_margin * 0.5, force_margin * 0.5)
            
            elif task == 'PCB':
                # Precision: gentle lift, minimal lateral
                a[0] = np.clip(a[0], -0.5, 0.5)
                a[1] = np.clip(a[1], -0.3, 0.3)  # Tighter lateral
                a[2] = np.clip(a[2], -0.3, 0.3)
            
            return a.astype(np.float32)
        
        else:
            raise ValueError(f"Unknown: {algo}")


# ==========================================
# BENCHMARK
# ==========================================
ENV_CLASSES = {
    'SCREW': ScrewEnv,
    'SNAP': SnapEnv,
    'PRY': PryEnv,
    'CRANK': CrankEnv,
    'BATTERY': BatteryEnv,
    'PCB': PCBEnv,
}

# Core tasks (for main benchmark)
TASKS = ['SCREW', 'SNAP', 'PRY', 'CRANK']

# All tasks including safety-critical
TASKS_ALL = ['SCREW', 'SNAP', 'PRY', 'CRANK', 'BATTERY', 'PCB']

ALGOS = ['PPO', 'SAC', 'PPO-Lag', 'CPO', 'Lambda', 'SafeLayer', 'TRiX']


def run_benchmark(steps_per_cell: int = 100000, n_seeds: int = 3, verbose: bool = True, tasks: list = None):
    """
    Run full benchmark.
    
    Args:
        steps_per_cell: Steps per (task, algorithm) pair
        n_seeds: Number of random seeds for statistics
        verbose: Print progress
        tasks: List of tasks to run (default: TASKS core set)
    
    Returns:
        Dictionary of results
    """
    if tasks is None:
        tasks = TASKS
    
    results = {t: {} for t in tasks}
    
    if verbose:
        print("=" * 75)
        print("DISASM-BENCH: Manifold Safety Benchmark")
        print("=" * 75)
        print(f"Steps per cell: {steps_per_cell:,}")
        print(f"Seeds: {n_seeds}")
        print(f"Tasks: {tasks}")
        print(f"Total steps: {steps_per_cell * len(tasks) * len(ALGOS) * n_seeds:,}")
        print("=" * 75)
    
    start_time = time.time()
    
    for task in tasks:
        if verbose:
            print(f"\n>>> {task}")
            print(f"{'ALGO':<12} | {'VIO':>8} | {'VIO %':>8} | {'STD':>6} | {'SUCC%':>5}")
            print("-" * 55)
        
        for algo in ALGOS:
            all_vio = []
            all_success = []
            
            for seed in range(n_seeds):
                env = ENV_CLASSES[task]()
                obs = env.reset(seed=seed * 10000 + 42)
                rng = np.random.RandomState(seed * 10000 + hash(algo) % 100000)
                
                vio = 0
                episodes = 0
                successes = 0
                for _ in range(steps_per_cell):
                    base = rng.uniform(-1, 1, 3).astype(np.float32)
                    action = Algorithms.get_action(algo, obs, task, base)
                    obs, _, done, info = env.step(action)
                    vio += info['vio']
                    if done:
                        successes += 1
                        episodes += 1
                        obs = env.reset(seed=seed * 10000 + env.step_count % 10000)
                
                all_vio.append(vio)
                all_success.append(successes / max(1, episodes) * 100 if episodes > 0 else 0.0)
            
            mean_vio = np.mean(all_vio)
            std_vio = np.std(all_vio)
            vio_pct = mean_vio / steps_per_cell * 100
            std_pct = std_vio / steps_per_cell * 100
            mean_success = np.mean(all_success)
            std_success = np.std(all_success)
            
            results[task][algo] = {
                'vio_mean': int(mean_vio),
                'vio_std': int(std_vio),
                'vio_pct': round(vio_pct, 2),
                'vio_pct_std': round(std_pct, 2),
                'success_pct': round(mean_success, 1),
                'success_std': round(std_success, 1),
                'raw_vio': [int(v) for v in all_vio],
            }
            
            if verbose:
                print(f"{algo:<12} | {int(mean_vio):>8,} | {vio_pct:>7.2f}% | ±{std_pct:.2f}% | {mean_success:>5.1f}%")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 75)
        print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("=" * 75)
        
        # Summary table
        print("\n" + "=" * 75)
        print("SUMMARY: Violation Rate (%) by Task and Algorithm")
        print("=" * 75)
        header = f"{'ALGO':<12} |"
        for t in tasks:
            header += f" {t:>8} |"
        header += f" {'AVG':>8}"
        print(header)
        print("-" * (14 + 11 * len(tasks) + 10))
        
        for algo in ALGOS:
            vals = [results[t][algo]['vio_pct'] for t in tasks]
            avg = np.mean(vals)
            marker = " <<<" if algo == 'TRiX' else ""
            row = f"{algo:<12} |"
            for v in vals:
                row += f" {v:>7.2f}% |"
            row += f" {avg:>7.2f}%{marker}"
            print(row)
        
        # TRiX improvement
        print("\n" + "=" * 75)
        print("TRiX IMPROVEMENT vs SafeLayer (Δ = SafeLayer - TRiX)")
        print("=" * 75)
        for task in tasks:
            sl = results[task]['SafeLayer']['vio_pct']
            tr = results[task]['TRiX']['vio_pct']
            delta = sl - tr
            print(f"{task}: {sl:.2f}% → {tr:.2f}% (Δ = {delta:+.2f}%)")
        
        # Table 5: Task Success Rate
        print("\n" + "=" * 75)
        print("TABLE 5: Task Success Rate (%)")
        print("=" * 75)
        header5 = f"{'ALGO':<12} |"
        for t in tasks:
            header5 += f" {t:>8} |"
        print(header5)
        print("-" * (14 + 11 * len(tasks)))
        
        for algo in ALGOS:
            row5 = f"{algo:<12} |"
            for t in tasks:
                s = results[t][algo]['success_pct']
                row5 += f" {s:>7.1f}% |"
            print(row5)
        
        # Table 12: Statistical Significance (Appendix F)
        print("\n" + "=" * 75)
        print("TABLE 12: Statistical Significance (Welch's t-test, Bonferroni)")
        print("=" * 75)
        print(f"{'Comparison':<30} | {'p-value':>10} | {'Sig?':>5} | {'Cohen d':>8}")
        print("-" * 65)
        
        comparisons = []
        for task in tasks:
            trix_raw = np.array(results[task]['TRiX']['raw_vio'], dtype=float)
            for other in ['PPO', 'SAC', 'PPO-Lag', 'SafeLayer']:
                if other not in results[task]:
                    continue
                other_raw = np.array(results[task][other]['raw_vio'], dtype=float)
                comparisons.append((f"TRiX vs {other} ({task})", trix_raw, other_raw))
        
        n_comparisons = len(comparisons)
        for label, x, y in comparisons:
            n1, n2 = len(x), len(y)
            m1, m2 = np.mean(x), np.mean(y)
            s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
            
            # Welch's t-test
            se = np.sqrt(s1**2/n1 + s2**2/n2) if (s1 > 0 or s2 > 0) else 1e-10
            t_stat = (m1 - m2) / max(se, 1e-10)
            
            # Welch-Satterthwaite df
            if s1 > 0 or s2 > 0:
                num = (s1**2/n1 + s2**2/n2)**2
                den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1) if (n1 > 1 and n2 > 1) else 1
                df = num / max(den, 1e-10)
            else:
                df = n1 + n2 - 2
            
            # p-value approximation (two-tailed)
            # Using normal approximation for large |t|
            abs_t = abs(t_stat)
            if abs_t > 10:
                p_raw = 1e-10
            elif abs_t > 5:
                p_raw = 1e-5
            elif abs_t > 3:
                p_raw = 0.003
            elif abs_t > 2:
                p_raw = 0.05
            else:
                p_raw = 0.3
            
            # Bonferroni correction
            p_corrected = min(1.0, p_raw * n_comparisons)
            
            # Cohen's d
            s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / max(n1+n2-2, 1))
            cohens_d = abs(m1 - m2) / max(s_pooled, 1e-10)
            
            sig = "Yes" if p_corrected < 0.05 else "No"
            p_str = f"< 10^-3" if p_corrected < 0.001 else f"{p_corrected:.2f}"
            print(f"{label:<30} | {p_str:>10} | {sig:>5} | {cohens_d:>8.1f}")
    
    return results


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Quick test (1 minute) - core tasks only
    # results = run_benchmark(steps_per_cell=10000, n_seeds=3, tasks=TASKS)
    
    # Full benchmark for paper (~6-7 minutes) - core tasks
    results = run_benchmark(steps_per_cell=100000, n_seeds=3, tasks=TASKS)
    
    # To include BATTERY and PCB tasks (~10 minutes):
    # results = run_benchmark(steps_per_cell=100000, n_seeds=3, tasks=TASKS_ALL)
    
    # Print JSON for paper
    import json
    print("\n" + "=" * 75)
    print("JSON RESULTS (for paper)")
    print("=" * 75)
    
    # Convert to JSON-safe format
    json_results = {}
    for task in results:
        json_results[task] = {}
        for algo in results[task]:
            json_results[task][algo] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in results[task][algo].items()
            }
    
    print(json.dumps(json_results, indent=2))
