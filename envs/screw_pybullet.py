"""
PyBullet-based screw removal environment with realistic physics.

This replaces the abstract "screw proxy" with actual:
- 7-DoF robot arm (Panda)
- Screw geometry with threads
- Contact physics (friction, jamming)
- Force/torque sensing
- Vision-based symbolic state extraction

Drop-in replacement for screw_proxy.py but with real physics.
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Tuple, Optional
import os

from .base_env import BaseDisassemblyEnv, EntropyFactor


class ScrewPyBulletEnv(BaseDisassemblyEnv):
    """
    Physics-based screw removal using PyBullet.
    
    State:
        - Robot joint angles (7 DoF)
        - End-effector pose
        - Contact forces (from physics)
        - Screw state (position, removal progress)
    
    Actions:
        - End-effector velocity commands (6D: xyz + rpy)
    
    Success:
        - Screw extracted (z-position threshold reached)
    """
    
    def _init_task_specific(self):
        """Initialize PyBullet and screw-specific parameters."""
        # PyBullet setup
        self.physics_client = None
        self.robot_id = None
        self.screw_id = None
        self.table_id = None
        self.tool_id = None
        
        # Task parameters
        self.screw_position_nominal = np.array([0.5, 0.0, 0.05])  # On table
        self.screw_orientation = 0.0
        self.screw_present = True
        self.screw_stuck_factor = 0.0
        self.screw_threads = 10  # Number of thread rotations
        
        # Physics parameters
        self.screw_friction = 0.8
        self.screw_mass = 0.005  # 5 grams
        self.contact_stiffness = 1000.0
        self.contact_damping = 10.0
        
        # Removal tracking
        self.removal_progress = 0.0  # 0 to 1
        self.applied_torque_history = []
        self.contact_history = []
        
        # Constraints
        self.alignment_tolerance = 0.003  # 3mm
        self.max_torque = 0.8  # Nm
        self.max_extraction_force = 50.0  # N
        
        # Hazard zone (battery nearby)
        self.hazard_zone = np.array([0.45, 0.55, -0.1, 0.1, 0.0, 0.15])
        
        # Entropy factors
        self.active_entropy = []
        
        # Rendering
        self.render_mode = None
        
    def get_symbolic_state_dim(self) -> int:
        """Symbolic predicates from physics."""
        return 6  # [screw_present, aligned, contact, in_hazard, high_force, rotation_correct]
    
    def _init_physics(self):
        """Initialize PyBullet physics simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Connect to PyBullet
        self.physics_client = p.connect(p.DIRECT)  # No GUI for training
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set gravity
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(0.01, physicsClientId=self.physics_client)  # 100 Hz
        
        # Load plane and table
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0.0, 0.0],
            physicsClientId=self.physics_client
        )
        
        # Load Panda robot arm
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.physics_client
        )
        
        # Create screw (cylinder with thread approximation)
        self._create_screw()
        
        # Create screwdriver tool
        self._create_tool()
        
        # Initialize robot to home position
        self._reset_robot()
        
        # Enable force/torque sensing
        p.enableJointForceTorqueSensor(
            self.robot_id,
            6,  # End-effector joint
            enableSensor=True,
            physicsClientId=self.physics_client
        )
        
    def _create_screw(self):
        """Create screw object with realistic properties."""
        # Screw dimensions (M4 screw)
        screw_radius = 0.002  # 2mm radius
        screw_length = 0.02   # 20mm length
        screw_head_radius = 0.0035  # 3.5mm
        screw_head_height = 0.003  # 3mm
        
        # Create collision shape (simplified as cylinder)
        screw_shaft_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=screw_radius,
            height=screw_length,
            physicsClientId=self.physics_client
        )
        
        screw_head_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=screw_head_radius,
            height=screw_head_height,
            physicsClientId=self.physics_client
        )
        
        # Visual shape
        screw_shaft_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=screw_radius,
            length=screw_length,
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=self.physics_client
        )
        
        screw_head_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=screw_head_radius,
            length=screw_head_height,
            rgbaColor=[0.6, 0.6, 0.6, 1],
            physicsClientId=self.physics_client
        )
        
        # Create multi-body
        self.screw_id = p.createMultiBody(
            baseMass=self.screw_mass,
            baseCollisionShapeIndex=screw_shaft_col,
            baseVisualShapeIndex=screw_shaft_vis,
            basePosition=self.screw_position_nominal,
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            physicsClientId=self.physics_client
        )
        
        # Set friction
        p.changeDynamics(
            self.screw_id,
            -1,
            lateralFriction=self.screw_friction,
            spinningFriction=0.001,
            rollingFriction=0.001,
            physicsClientId=self.physics_client
        )
        
    def _create_tool(self):
        """Create screwdriver tool (simplified as cylinder)."""
        tool_radius = 0.0015  # 1.5mm (Phillips bit)
        tool_length = 0.05    # 50mm
        
        tool_col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=tool_radius,
            height=tool_length,
            physicsClientId=self.physics_client
        )
        
        tool_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=tool_radius,
            length=tool_length,
            rgbaColor=[0.2, 0.2, 0.8, 1],
            physicsClientId=self.physics_client
        )
        
        self.tool_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=tool_col,
            baseVisualShapeIndex=tool_vis,
            basePosition=[0.5, 0, 0.2],
            physicsClientId=self.physics_client
        )
        
        # Attach tool to robot end-effector (constraint)
        self.tool_constraint = p.createConstraint(
            self.robot_id,
            8,  # End-effector link
            self.tool_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.05],  # Tool offset from EE
            [0, 0, 0],
            physicsClientId=self.physics_client
        )
        
    def _reset_robot(self):
        """Reset robot to home position above screw."""
        # Home joint angles (above table)
        home_joints = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]
        
        for i in range(7):
            p.resetJointState(
                self.robot_id,
                i,
                home_joints[i],
                physicsClientId=self.physics_client
            )
        
    def extract_symbolic_state(self) -> np.ndarray:
        """
        Extract symbolic predicates from physics state.
        
        Returns:
            [screw_present, aligned, contact, in_hazard, high_force, rotation_correct]
        """
        if not self.screw_present:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Get tool pose
        tool_pos, tool_orn = p.getBasePositionAndOrientation(
            self.tool_id,
            physicsClientId=self.physics_client
        )
        tool_pos = np.array(tool_pos)
        
        # Get screw pose
        screw_pos, screw_orn = p.getBasePositionAndOrientation(
            self.screw_id,
            physicsClientId=self.physics_client
        )
        screw_pos = np.array(screw_pos)
        
        # 1. Screw present (can be occluded or missing)
        screw_present = float(self.screw_present)
        
        # 2. Alignment: tool tip near screw head
        alignment_error = np.linalg.norm(tool_pos[:2] - screw_pos[:2])
        aligned = float(alignment_error < self.alignment_tolerance)
        
        # 3. Contact: check contact points
        contact_points = p.getContactPoints(
            bodyA=self.tool_id,
            bodyB=self.screw_id,
            physicsClientId=self.physics_client
        )
        in_contact = float(len(contact_points) > 0)
        
        # 4. In hazard zone
        in_hazard = float(self._in_hazard_zone(tool_pos))
        
        # 5. High force: from contact points
        total_force = 0.0
        if contact_points:
            total_force = sum(cp[9] for cp in contact_points)  # Normal force
        high_force = float(total_force > 30.0)
        
        # 6. Rotation correct: tool aligned with screw axis
        tool_euler = p.getEulerFromQuaternion(tool_orn)
        screw_euler = p.getEulerFromQuaternion(screw_orn)
        rotation_error = abs(tool_euler[2] - screw_euler[2])
        rotation_correct = float(rotation_error < 0.1)  # ~6 degrees
        
        return np.array([
            screw_present,
            aligned,
            in_contact,
            in_hazard,
            high_force,
            rotation_correct
        ], dtype=np.float32)
        
    def apply_procedural_entropy(self):
        """Apply entropy factors to screw configuration."""
        rng = self._np_random
        
        # Missing screw
        if rng.random() < self.entropy_level * 0.1:
            self.screw_present = False
            self.active_entropy.append(EntropyFactor.MISSING_COMPONENT)
            # Remove screw from simulation
            if self.screw_id is not None:
                p.removeBody(self.screw_id, physicsClientId=self.physics_client)
            return
        
        # Position offset
        if rng.random() < self.entropy_level:
            offset = rng.randn(3) * 0.01 * self.entropy_level
            self.screw_position_nominal += offset
            self.active_entropy.append(EntropyFactor.TOLERANCE_SHIFT)
        
        # Stuck screw (increased friction)
        if rng.random() < self.entropy_level:
            self.screw_stuck_factor = rng.random() * self.entropy_level
            # Increase friction in physics
            friction_mult = 1.0 + self.screw_stuck_factor * 3.0
            p.changeDynamics(
                self.screw_id,
                -1,
                lateralFriction=self.screw_friction * friction_mult,
                spinningFriction=0.001 * friction_mult,
                physicsClientId=self.physics_client
            )
            self.active_entropy.append(EntropyFactor.FRICTION_INCREASE)
        
        # Damaged threads (reduced thread depth)
        if rng.random() < self.entropy_level * 0.5:
            self.screw_threads = int(self.screw_threads * (1.0 - 0.3 * rng.random()))
            self.active_entropy.append(EntropyFactor.DAMAGE_OFFSET)
        
    def _step_physics(self, action: np.ndarray):
        """
        Apply action and step physics simulation.
        
        Action: [vx, vy, vz, wx, wy, wz] (end-effector velocities)
        """
        dt = 0.01  # 100 Hz
        
        # Get current end-effector state
        ee_state = p.getLinkState(
            self.robot_id,
            8,  # End-effector link index
            computeLinkVelocity=1,
            physicsClientId=self.physics_client
        )
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        # Compute target position/orientation
        target_pos = ee_pos + action[:3] * dt
        
        # Convert angular velocity to orientation change
        ee_euler = p.getEulerFromQuaternion(ee_orn)
        target_euler = ee_euler + action[3:] * dt
        target_orn = p.getQuaternionFromEuler(target_euler)
        
        # Inverse kinematics to get joint targets
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            8,  # End-effector link
            target_pos,
            target_orn,
            physicsClientId=self.physics_client
        )
        
        # Set joint targets
        p.setJointMotorControlArray(
            self.robot_id,
            range(7),
            p.POSITION_CONTROL,
            targetPositions=joint_targets[:7],
            forces=[87] * 7,  # Max force per joint
            physicsClientId=self.physics_client
        )
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # Update removal progress based on screw motion
        if self.screw_present:
            self._update_screw_extraction()
        
        # Update state dict
        self._update_state_from_physics()
        
    def _update_screw_extraction(self):
        """Update screw removal progress based on physics."""
        if self.screw_id is None:
            return
        
        # Get screw position
        screw_pos, _ = p.getBasePositionAndOrientation(
            self.screw_id,
            physicsClientId=self.physics_client
        )
        
        # Check if screw moved upward (extraction)
        z_displacement = screw_pos[2] - self.screw_position_nominal[2]
        
        # Removal progress based on vertical displacement
        # Full extraction = 15mm above original position
        self.removal_progress = np.clip(z_displacement / 0.015, 0.0, 1.0)
        
    def _update_state_from_physics(self):
        """Update self.state dict from PyBullet physics state."""
        # Joint states
        joint_states = [p.getJointState(self.robot_id, i, physicsClientId=self.physics_client) 
                       for i in range(7)]
        self.state['q'] = np.array([js[0] for js in joint_states])
        self.state['dq'] = np.array([js[1] for js in joint_states])
        
        # End-effector state
        ee_state = p.getLinkState(
            self.robot_id, 8,
            computeLinkVelocity=1,
            physicsClientId=self.physics_client
        )
        ee_pos = np.array(ee_state[0])
        ee_orn = p.getEulerFromQuaternion(ee_state[1])
        self.state['ee_pose'] = np.concatenate([ee_pos, ee_orn])
        
        # Contact forces
        contact_points = p.getContactPoints(
            bodyA=self.tool_id,
            bodyB=self.screw_id,
            physicsClientId=self.physics_client
        )
        
        total_force = np.zeros(3)
        if contact_points:
            for cp in contact_points:
                normal_force = cp[9]
                contact_normal = np.array(cp[7])
                total_force += normal_force * contact_normal
        
        self.state['contact_force'] = total_force
        
    def _compute_reward(self) -> Tuple[float, Dict]:
        """Compute task reward from physics state."""
        reward = 0.0
        info = {}
        
        if not self.screw_present:
            reward = -0.5
            info['missing_screw_penalty'] = True
            return reward, info
        
        # Progress reward
        reward += self.removal_progress * 20.0
        
        # Alignment reward
        symbolic = self.extract_symbolic_state()
        if symbolic[1] > 0.5:  # aligned
            reward += 2.0
            info['aligned'] = True
        
        # Contact reward (when aligned)
        if symbolic[1] > 0.5 and symbolic[2] > 0.5:  # aligned and in contact
            reward += 3.0
            info['proper_contact'] = True
        
        # Penalize excessive force
        force_mag = np.linalg.norm(self.state['contact_force'])
        if force_mag > self.max_extraction_force:
            reward -= (force_mag - self.max_extraction_force) * 0.1
        
        info['progress'] = self.removal_progress
        info['force_magnitude'] = force_mag
        
        return reward, info
        
    def _check_success(self) -> bool:
        """Success if screw is fully extracted."""
        return self.screw_present and self.removal_progress >= 0.95
        
    def close(self):
        """Cleanup PyBullet."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Factory function
def make_screw_pybullet_env(entropy_level=0.3, **kwargs):
    """Create PyBullet-based screw environment."""
    return ScrewPyBulletEnv(entropy_level=entropy_level, control_mode="ee_velocity", **kwargs)
