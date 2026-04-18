"""
rl_agent.py
===========
Composants Deep Q-Learning pour la baseline RL.

Classes :
  DQN          : réseau Q-value (MLP configurable)
  ReplayBuffer : buffer d'expérience (numpy pour performance)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections import deque
import random


class DQN(nn.Module):
    """
    Réseau Q-value avec couches cachées configurables.

    Args:
        state_dim   : dimension de l'état (nombre de features)
        hidden_dims : liste des dimensions des couches cachées
        n_actions   : nombre d'actions (3 : sell, hold, buy)
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dims: list[int] | None = None,
        n_actions: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """
    Buffer d'expérience circulaire pour le DQN.

    Stocke (state, action, reward, next_state, done) en numpy
    pour des performances optimales lors du sampling.
    """
    def __init__(self, capacity: int = 10_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype="float32"),
            np.array(actions,     dtype="int64"),
            np.array(rewards,     dtype="float32"),
            np.array(next_states, dtype="float32"),
            np.array(dones,       dtype="float32"),
        )

    def __len__(self) -> int:
        return len(self.buffer)
