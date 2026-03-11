"""
Screw Proxy Task for DISASM-Bench

Task: Align tool with screw head and apply torque to remove screw
Success: Torque threshold reached (screw fully removed)
Entropy: Missing screw, position offset, occlusion

Symbolic constraints:
- Must align tool before applying high torque
- Must not exceed torque limit
- Must not apply force in wrong direction
"""

import numpy as np
from typing import Dict, Tuple
from .base_env import BaseDisassemblyEnv, EntropyFactor


class ScrewProxyEnv(BaseDisassemblyEnv):
    """
    Screw removal proxy task.
    
    State:
        - Screw position (xyz)
        - Screw orientation (angle)
        - Tool position (xyz)
        - Tool orientation (angle)
        - Alignment error
        - Applied torque
        - Removal progress (0 to 1)
    
    Actions:
        - Tool position adjustment (xyz)
        - Tool rotation (1D)
        - Torque application (1D)
    
    Success:
        removal_progress >= 1.0
    """
    
    def _init_task_specific(self):
        """Initialize screw-specific parameters."""
        # Nominal screw properties
        self.screw_position = np.array([0.3, 0.0, 0.1])  # xyz in meters
        self.screw_orientation = 0.0  # radians
        self.screw_present = True
        self.screw_stuck_factor = 0.0  # 0 = easy, 1 = very stuck
        
        # Task parameters
        self.alignment_tolerance = 0.002  # 2mm radial tolerance
        self.torque_threshold = 0.5  # Nm for removal
        self.max_torque = 1.5  # Nm maximum safe torque
        self.removal_progress = 0.0
        
        # Hazard zone (battery region nearby)
        self.hazard_zone = np.array([0.25, 0.35, -0.05, 0.05, 0.0, 0.15])
        
        # Entropy factors applied
        self.active_entropy = []
        
    def get_symbolic_state_dim(self) -> int:
        """Symbolic state: [screw_present, aligned, in_hazard, high_force]"""
        return 4
        
    def extract_symbolic_state(self) -> np.ndarray:
        """Extract binary predicates."""
        tool_pos = self.state['ee_pose'][:3]
        
        # Alignment check
        alignment_error = np.linalg.norm(tool_pos[:2] - self.screw_position[:2])
        aligned = float(alignment_error < self.alignment_tolerance)
        
        # Hazard proximity
        in_hazard = float(self._in_hazard_zone(tool_pos))
        
        # Force level
        force_mag = np.linalg.norm(self.state['contact_force'])
        high_force = float(force_mag > 30.0)
        
        return np.array([
            float(self.screw_present),
            aligned,
            in_hazard,
            high_force
        ], dtype=np.float32)
        
    def apply_procedural_entropy(self):
        """Apply entropy factors based on entropy_level."""
        rng = np.random.RandomState(self.np_random.integers(0, 1000000))
        
        # Determine which entropy factors to apply
        if rng.random() < self.entropy_level:
            # Missing screw (10% chance at high entropy)
            if rng.random() < 0.1:
                self.screw_present = False
                self.active_entropy.append(EntropyFactor.MISSING_COMPONENT)
        
        if rng.random() < self.entropy_level:
            # Position offset (up to 5cm)
            offset = rng.randn(3) * 0.05 * self.entropy_level
            self.screw_position += offset
            self.active_entropy.append(EntropyFactor.TOLERANCE_SHIFT)
        
        if rng.random() < self.entropy_level:
            # Stuck screw (increased friction)
            self.screw_stuck_factor = rng.random() * self.entropy_level
            self.active_entropy.append(EntropyFactor.FRICTION_INCREASE)
        
        if rng.random() < self.entropy_level * 0.5:
            # Occlusion (nearby component blocking view)
            # For simulation, we model this as reduced alignment tolerance
            self.alignment_tolerance *= (1.0 + rng.random() * 0.5)
            self.active_entropy.append(EntropyFactor.OCCLUSION)
            
    def _update_object_dynamics(self):
        """Update screw removal dynamics."""
        if not self.screw_present:
            return
        
        tool_pos = self.state['ee_pose'][:3]
        alignment_error = np.linalg.norm(tool_pos[:2] - self.screw_position[:2])
        
        # Torque can only be applied if aligned
        if alignment_error < self.alignment_tolerance:
            # Get torque from action (last dimension if available)
            # For simplicity, infer torque from contact force z-component
            applied_torque = abs(self.state['contact_force'][2]) * 0.01
            
            # Progress depends on torque and stuck factor
            effective_torque = max(0, applied_torque - self.screw_stuck_factor * 0.3)
            self.removal_progress += effective_torque * 0.02
            
            # Cap at 1.0
            self.removal_progress = min(1.0, self.removal_progress)
        
    def _compute_reward(self) -> Tuple[float, Dict]:
        """Compute task reward."""
        reward = 0.0
        info = {}
        
        if not self.screw_present:
            # Special case: trying to remove missing screw (negative reward)
            reward = -0.1
            info['missing_screw_penalty'] = True
            return reward, info
        
        # Progress reward
        reward += self.removal_progress * 10.0
        
        # Alignment bonus
        tool_pos = self.state['ee_pose'][:3]
        alignment_error = np.linalg.norm(tool_pos[:2] - self.screw_position[:2])
        if alignment_error < self.alignment_tolerance:
            reward += 1.0
            info['aligned'] = True
        else:
            reward -= alignment_error * 5.0
        
        # Penalize excessive force
        force_mag = np.linalg.norm(self.state['contact_force'])
        if force_mag > self.max_torque * 10:  # convert to force proxy
            reward -= (force_mag - self.max_torque * 10) * 0.5
        
        info['progress'] = self.removal_progress
        info['alignment_error'] = alignment_error
        
        return reward, info
        
    def _compute_sequential_violations(self, state: Dict, action: np.ndarray) -> Tuple[float, Dict]:
        """Check sequential constraints."""
        cost = 0.0
        info = {}
        
        # Rule: Do not apply high torque without alignment
        tool_pos = state['ee_pose'][:3]
        alignment_error = np.linalg.norm(tool_pos[:2] - self.screw_position[:2])
        force_mag = np.linalg.norm(state['contact_force'])
        
        if alignment_error > self.alignment_tolerance and force_mag > 20.0:
            cost += 15.0
            info['misaligned_torque_violation'] = True
        
        # Rule: Do not exceed max torque
        if force_mag > self.max_torque * 10:
            cost += 20.0
            info['torque_limit_violation'] = True
        
        return cost, info
        
    def _check_success(self) -> bool:
        """Success if screw is fully removed."""
        return self.screw_present and self.removal_progress >= 1.0
        
    def _check_failure(self) -> bool:
        """Failure if trying to remove missing screw for too long."""
        if not self.screw_present and self.timestep > 50:
            return True
        return False


# Factory function
def make_screw_env(entropy_level=0.3, **kwargs):
    """Create ScrewProxyEnv with specified entropy."""
    return ScrewProxyEnv(entropy_level=entropy_level, **kwargs)
