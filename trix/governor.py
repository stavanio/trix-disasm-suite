"""
TRiX: Transparent Real-time eXplainable Control

Core implementation of neuro-symbolic gating with deterministic explanations.

Components:
1. Symbolic rule system (φ_t computation)
2. Smooth feasibility gating (sigmoid mask)
3. Neural policy (SAC-based)
4. Deterministic explanation generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Rule:
    """Symbolic safety rule."""
    rule_id: str
    weight: float
    description: str
    check_fn: callable  # z_t, u -> violation_value (≥0)
    explanation_template: str
    safe_alternative: str


class RuleLibrary:
    """Library of symbolic safety rules for disassembly tasks."""
    
    def __init__(self):
        self.rules: List[Rule] = []
        self._build_rules()
        
    def _build_rules(self):
        """Define standard disassembly safety rules."""
        
        # Rule 1: No high force without alignment
        self.rules.append(Rule(
            rule_id="R1_MISALIGNED_FORCE",
            weight=10.0,
            description="High force without proper alignment",
            check_fn=lambda z, u: max(0, (1.0 - z[1]) * np.linalg.norm(u[:3]) - 0.05),
            explanation_template="Tool not aligned with target. High force may damage component.",
            safe_alternative="Align tool before applying force (reduce velocity, adjust position)"
        ))
        
        # Rule 2: No high velocity in hazard zone
        self.rules.append(Rule(
            rule_id="R2_HAZARD_VELOCITY",
            weight=20.0,
            description="Excessive velocity near hazard zone",
            check_fn=lambda z, u: max(0, z[2] * (np.linalg.norm(u[:3]) - 0.03)),
            explanation_template="Approaching battery/hazard region too quickly.",
            safe_alternative="Reduce velocity to < 0.03 m/s near hazard zones"
        ))
        
        # Rule 3: No force on missing component
        self.rules.append(Rule(
            rule_id="R3_MISSING_COMPONENT",
            weight=15.0,
            description="Applying force to missing component",
            check_fn=lambda z, u: max(0, (1.0 - z[0]) * np.linalg.norm(u)),
            explanation_template="Target component not detected. Force application invalid.",
            safe_alternative="Verify component presence before applying force"
        ))
        
        # Rule 4: Force magnitude limit
        self.rules.append(Rule(
            rule_id="R4_FORCE_LIMIT",
            weight=15.0,
            description="Exceeding maximum safe force",
            check_fn=lambda z, u: max(0, np.linalg.norm(u) - 0.5),
            explanation_template="Action exceeds maximum safe force threshold.",
            safe_alternative="Reduce action magnitude to < 0.5"
        ))
        
    def compute_violation_score(self, z_t: np.ndarray, u: np.ndarray) -> Tuple[float, List[str]]:
        """
        Compute total violation score φ_t(u).
        
        Args:
            z_t: Symbolic state vector (predicates)
            u: Candidate action
            
        Returns:
            phi: Total violation score
            triggered_rules: List of rule IDs that were violated
        """
        phi = 0.0
        triggered_rules = []
        
        for rule in self.rules:
            violation = rule.check_fn(z_t, u)
            if violation > 0:
                phi += rule.weight * violation
                triggered_rules.append(rule.rule_id)
        
        return phi, triggered_rules
        
    def generate_explanation(self, z_t: np.ndarray, u: np.ndarray, phi: float) -> Dict:
        """
        Generate deterministic explanation from rule triggers.
        
        Returns dict with:
            - rule_id: Primary violated rule
            - description: Human-readable explanation
            - safe_alternative: Suggested safe action
            - confidence: 1.0 - sigmoid(phi - tau) 
            - risk_level: "low" | "medium" | "high"
        """
        _, triggered = self.compute_violation_score(z_t, u)
        
        if not triggered:
            return {
                'rule_id': None,
                'description': "Action within safety bounds",
                'safe_alternative': "Continue with proposed action",
                'confidence': 1.0,
                'risk_level': "low"
            }
        
        # Find most violated rule (highest weighted contribution)
        max_violation = 0
        primary_rule = None
        
        for rule in self.rules:
            if rule.rule_id in triggered:
                violation = rule.weight * rule.check_fn(z_t, u)
                if violation > max_violation:
                    max_violation = violation
                    primary_rule = rule
        
        # Risk level based on total phi
        if phi < 5.0:
            risk_level = "low"
        elif phi < 15.0:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            'rule_id': primary_rule.rule_id,
            'description': primary_rule.explanation_template,
            'safe_alternative': primary_rule.safe_alternative,
            'confidence': 1.0,  # Deterministic
            'risk_level': risk_level,
            'violation_score': phi,
            'triggered_rules': triggered
        }


class SmoothGate(nn.Module):
    """Smooth feasibility gate using sigmoid."""
    
    def __init__(self, tau: float = 5.0, eta: float = 2.0):
        """
        Args:
            tau: Violation threshold margin
            eta: Temperature (smoothness parameter)
        """
        super().__init__()
        self.tau = tau
        self.eta = eta
        
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth gating factor.
        
        Args:
            phi: Violation score (batch or scalar)
            
        Returns:
            s: Gating factor in [0, 1]
        """
        return torch.sigmoid((self.tau - phi) / self.eta)
        
    def numpy(self, phi: float) -> float:
        """NumPy version for deployment."""
        return 1.0 / (1.0 + np.exp(-(self.tau - phi) / self.eta))


class TRiXPolicy:
    """
    TRiX neuro-symbolic gated policy.
    
    π_TRiX(u|o_t) ∝ π_θ(u|o_t) · M̃_t(u)
    
    Where M̃_t(u) = sigmoid((τ - φ_t(u)) / η)
    """
    
    def __init__(
        self,
        neural_policy,  # Base SAC policy
        rule_library: RuleLibrary,
        fallback_controller,
        tau: float = 5.0,
        eta: float = 2.0,
        device: str = "cpu"
    ):
        self.neural_policy = neural_policy
        self.rules = rule_library
        self.fallback = fallback_controller
        self.gate = SmoothGate(tau, eta)
        self.device = device
        
        # Logging
        self.violation_history = []
        self.gating_history = []
        self.explanation_history = []
        
    def select_action(
        self,
        observation: np.ndarray,
        symbolic_state: np.ndarray,
        deterministic: bool = False,
        return_explanation: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Select action with TRiX gating.
        
        Args:
            observation: Full observation vector
            symbolic_state: Symbolic predicate vector z_t
            deterministic: Use mean action (for eval)
            return_explanation: Generate explanation
            
        Returns:
            action: Gated action u_t
            explanation: Explanation dict (if requested)
        """
        # Get neural proposal
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            u_neural = self.neural_policy.select_action(obs_tensor, deterministic=deterministic)
            
        # Get fallback (safe baseline)
        u_safe = self.fallback(observation, symbolic_state)
        
        # Compute violation score
        phi, triggered = self.rules.compute_violation_score(symbolic_state, u_neural)
        
        # Compute gating factor
        s = self.gate.numpy(phi)
        
        # Gated action: u_t = s·u_neural + (1-s)·u_safe
        action = s * u_neural + (1 - s) * u_safe
        
        # Log
        self.violation_history.append(phi)
        self.gating_history.append(s)
        
        # Generate explanation
        explanation = None
        if return_explanation:
            explanation = self.rules.generate_explanation(symbolic_state, u_neural, phi)
            explanation['gating_factor'] = s
            self.explanation_history.append(explanation)
        
        return action, explanation
        
    def update(self, replay_buffer, batch_size: int = 256) -> Dict:
        """
        Update neural policy with TRiX-aware training.
        
        The key modification: during training, sample actions and weight
        their Q-value updates by the gating factor to encourage the neural
        policy to propose feasible actions.
        """
        # Sample batch
        batch = replay_buffer.sample(batch_size)
        obs, actions, rewards, next_obs, dones, symbolic_states = batch
        
        # Standard SAC update, but we can add auxiliary loss
        # to encourage neural policy to avoid violations
        
        # Compute violation scores for actions in batch
        violations = []
        for i in range(batch_size):
            phi, _ = self.rules.compute_violation_score(symbolic_states[i], actions[i])
            violations.append(phi)
        violations = torch.FloatTensor(violations).to(self.device)
        
        # Add violation penalty to actor loss (encourages neural policy to be safer)
        violation_penalty = violations.mean() * 0.1
        
        # Update base SAC policy (implementation-specific)
        # ... standard SAC update code ...
        
        info = {
            'violation_penalty': violation_penalty.item(),
            'mean_gating': np.mean(self.gating_history[-100:]) if self.gating_history else 1.0
        }
        
        return info
        
    def get_metrics(self) -> Dict:
        """Get TRiX-specific metrics."""
        if not self.violation_history:
            return {}
        
        return {
            'mean_violation_score': np.mean(self.violation_history[-100:]),
            'max_violation_score': np.max(self.violation_history[-100:]),
            'mean_gating_factor': np.mean(self.gating_history[-100:]),
            'n_explanations': len(self.explanation_history),
            'chatter_events': self._compute_chatter()
        }
        
    def _compute_chatter(self, delta_threshold: float = 0.3) -> int:
        """Count chatter events (rapid gating changes)."""
        if len(self.gating_history) < 2:
            return 0
        
        gating_array = np.array(self.gating_history[-200:])
        deltas = np.abs(np.diff(gating_array))
        return np.sum(deltas > delta_threshold)


class SafeFallbackController:
    """Simple safe fallback controller (e.g., hold position, move to safe zone)."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        
    def __call__(self, observation: np.ndarray, symbolic_state: np.ndarray) -> np.ndarray:
        """
        Generate safe fallback action.
        
        For now: zero velocity (hold position)
        Can be enhanced with:
            - Move away from hazard zone
            - Retract to home position
            - Gentle approach
        """
        # Simple: zero action
        action = np.zeros(self.action_dim)
        
        # If in hazard zone (z[2] = 1), move away
        if symbolic_state[2] > 0.5:  # in_hazard predicate
            # Simple: move in -x direction (away from hazard)
            action[0] = -0.05
        
        return action


# ========== Training Integration ==========

def create_trix_agent(env, neural_policy_class, device="cpu"):
    """
    Create complete TRiX agent.
    
    Args:
        env: DISASM-Bench environment
        neural_policy_class: SAC or similar
        device: torch device
        
    Returns:
        TRiXPolicy instance
    """
    # Initialize neural policy (SAC)
    neural_policy = neural_policy_class(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device
    )
    
    # Initialize rule library
    rules = RuleLibrary()
    
    # Initialize fallback
    fallback = SafeFallbackController(action_dim=env.action_space.shape[0])
    
    # Create TRiX policy
    trix_policy = TRiXPolicy(
        neural_policy=neural_policy,
        rule_library=rules,
        fallback_controller=fallback,
        tau=5.0,
        eta=2.0,
        device=device
    )
    
    return trix_policy
