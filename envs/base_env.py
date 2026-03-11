"""
Base environment for DISASM-Bench disassembly tasks.

Defines common structure for all task families:
- Observation/action spaces
- Procedural entropy generation
- Safety cost computation
- Sequential constraint tracking
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class EntropyFactor(Enum):
    """Procedural entropy factors for disassembly tasks."""
    MISSING_COMPONENT = "missing"
    TOLERANCE_SHIFT = "tolerance_shift"
    DAMAGE_OFFSET = "damage"
    OCCLUSION = "occlusion"
    FRICTION_INCREASE = "stuck"
    BRITTLE_FAILURE = "brittle"
    PARTIAL_ADHESION = "partial_adhesion"


@dataclass
class SafetyConstraint:
    """Safety constraint specification."""
    name: str
    threshold: float
    cost_weight: float
    hazard_region: Optional[np.ndarray] = None  # [x_min, x_max, y_min, y_max, z_min, z_max]


class BaseDisassemblyEnv(gym.Env):
    """
    Abstract base class for DISASM-Bench environments.
    
    Observation space:
        - Joint positions/velocities (7 DoF arm)
        - End-effector pose (6D)
        - Contact forces (3D)
        - Symbolic state vector (predicates)
        - Vision features (optional, stubbed for now)
    
    Action space:
        - Joint velocity commands (7D) OR
        - End-effector velocity (6D: position + orientation)
    
    Reward:
        r_t = progress_reward - safety_cost
    
    Safety costs:
        - Force limit violations
        - Velocity in hazard zones
        - Sequential constraint violations
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        entropy_level: float = 0.3,  # 0.0 = nominal, 1.0 = maximum entropy
        use_vision: bool = False,
        control_mode: str = "joint_velocity",  # or "ee_velocity"
        max_episode_steps: int = 200,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.entropy_level = entropy_level
        self.use_vision = use_vision
        self.control_mode = control_mode
        self.max_episode_steps = max_episode_steps
        
        if seed is not None:
            self.seed(seed)
        
        # Environment state
        self.timestep = 0
        self.done = False
        
        # Task-specific initialization (override in subclasses)
        self._init_task_specific()
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Initialize physics (stub for now, can integrate PyBullet/MuJoCo)
        self._init_physics()
        
    def _init_task_specific(self):
        """Override in subclasses to set task-specific parameters."""
        raise NotImplementedError
        
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Action space: joint velocities or ee velocities
        if self.control_mode == "joint_velocity":
            self.action_dim = 7
            action_low = -np.ones(self.action_dim) * 0.5  # rad/s
            action_high = np.ones(self.action_dim) * 0.5
        else:  # ee_velocity
            self.action_dim = 6  # 3 linear + 3 angular
            action_low = np.array([-0.1, -0.1, -0.1, -0.5, -0.5, -0.5])
            action_high = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Observation space components
        obs_dim = (
            7 +      # joint positions
            7 +      # joint velocities
            6 +      # ee pose (xyz + rpy)
            3 +      # contact forces
            self.get_symbolic_state_dim() +  # symbolic predicates
            (128 if self.use_vision else 0)  # vision features (stub)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def _init_physics(self):
        """Initialize physics simulation (stub)."""
        # For now, use simple dynamics model
        # Can be replaced with PyBullet/MuJoCo client
        self.state = {
            'q': np.zeros(7),      # joint positions
            'dq': np.zeros(7),     # joint velocities
            'ee_pose': np.zeros(6),  # end-effector pose
            'contact_force': np.zeros(3),
            'object_state': {},    # task-specific object state
        }
        
    def get_symbolic_state_dim(self) -> int:
        """Return dimension of symbolic state vector."""
        # Override in subclasses
        return 0
        
    def extract_symbolic_state(self) -> np.ndarray:
        """
        Extract symbolic predicates from continuous state.
        
        Examples:
            - fastener_present: bool
            - distance_to_hazard: float
            - latch_engaged: bool
            - contact_detected: bool
        
        Returns:
            Binary/continuous predicate vector
        """
        raise NotImplementedError
        
    def apply_procedural_entropy(self):
        """Apply entropy factors to nominal task configuration."""
        raise NotImplementedError
        
    def compute_safety_cost(self, state: Dict, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute safety cost c_t based on violations.
        
        Returns:
            cost: float, total safety cost
            info: dict with individual violation components
        """
        cost = 0.0
        info = {}
        
        # Force limit violation
        force_magnitude = np.linalg.norm(state['contact_force'])
        force_limit = getattr(self, 'force_limit', 50.0)  # Newtons
        if force_magnitude > force_limit:
            force_violation = (force_magnitude - force_limit) / force_limit
            cost += 10.0 * force_violation
            info['force_violation'] = force_violation
        
        # Velocity in hazard zone
        if hasattr(self, 'hazard_zone'):
            ee_pos = state['ee_pose'][:3]
            if self._in_hazard_zone(ee_pos):
                velocity = np.linalg.norm(state['dq'][:3])  # simplified
                if velocity > 0.05:  # m/s threshold
                    cost += 20.0 * velocity
                    info['hazard_velocity_violation'] = velocity
        
        # Sequential constraint violations (task-specific)
        sequential_cost, sequential_info = self._compute_sequential_violations(state, action)
        cost += sequential_cost
        info.update(sequential_info)
        
        return cost, info
        
    def _in_hazard_zone(self, position: np.ndarray) -> bool:
        """Check if position is in defined hazard zone."""
        if not hasattr(self, 'hazard_zone') or self.hazard_zone is None:
            return False
        hz = self.hazard_zone
        return (hz[0] <= position[0] <= hz[1] and
                hz[2] <= position[1] <= hz[3] and
                hz[4] <= position[2] <= hz[5])
        
    def _compute_sequential_violations(self, state: Dict, action: np.ndarray) -> Tuple[float, Dict]:
        """Compute violations of sequential constraints (override in subclasses)."""
        return 0.0, {}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.timestep = 0
        self.done = False
        
        # Reset physics state
        self._init_physics()
        
        # Apply procedural entropy
        self.apply_procedural_entropy()
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action to physics (simplified dynamics)
        self._step_physics(action)
        
        # Compute reward and safety cost
        reward, reward_info = self._compute_reward()
        safety_cost, safety_info = self.compute_safety_cost(self.state, action)
        
        # Total reward
        total_reward = reward - safety_cost
        
        # Check termination
        self.timestep += 1
        success = self._check_success()
        truncated = self.timestep >= self.max_episode_steps
        terminated = success or self._check_failure()
        
        # Get observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'success': success,
            'safety_cost': safety_cost,
            'task_reward': reward,
            'timestep': self.timestep,
            **reward_info,
            **safety_info
        }
        
        return obs, total_reward, terminated, truncated, info
        
    def _step_physics(self, action: np.ndarray):
        """Update physics state (simplified, replace with PyBullet/MuJoCo)."""
        dt = 0.05  # 20 Hz control
        
        # Simple integration
        self.state['dq'] = action[:7] if len(action) == 7 else np.zeros(7)
        self.state['q'] += self.state['dq'] * dt
        
        # Update end-effector pose (simplified FK)
        # In real implementation, use robot kinematics
        self.state['ee_pose'][:3] += action[:3] * dt if len(action) == 6 else np.zeros(3)
        
        # Update task-specific object dynamics
        self._update_object_dynamics()
        
    def _update_object_dynamics(self):
        """Update object-specific dynamics (override in subclasses)."""
        pass
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs_components = [
            self.state['q'],
            self.state['dq'],
            self.state['ee_pose'],
            self.state['contact_force'],
            self.extract_symbolic_state(),
        ]
        
        if self.use_vision:
            # Stub vision features
            obs_components.append(np.random.randn(128) * 0.01)
        
        return np.concatenate(obs_components).astype(np.float32)
        
    def _compute_reward(self) -> Tuple[float, Dict]:
        """Compute task reward (override in subclasses)."""
        raise NotImplementedError
        
    def _check_success(self) -> bool:
        """Check if task is successfully completed."""
        raise NotImplementedError
        
    def _check_failure(self) -> bool:
        """Check if task has failed catastrophically."""
        # Default: no explicit failure condition
        return False
        
    def render(self):
        """Render environment (stub)."""
        pass
        
    def close(self):
        """Cleanup resources."""
        pass
