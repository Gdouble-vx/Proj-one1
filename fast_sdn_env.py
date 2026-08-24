"""
fast_sdn_env.py — FastSDNEnv (gymnasium wrapper รอบ NetworkSimulator)

จำลองเครือข่ายแบบเร็วมากสำหรับเทรน PPO (หลายพัน step/วินาที)
จากนั้นค่อยสลับไปเทรนต่อกับ ONOS จริงผ่าน --env onos (CustomSDNEnv)

Observation: node_feat(14) + edge_attr(50*2)  → 114 มิติ (raw graph state)
Action:      link weights 50 มิติ ในช่วง [1, 100]
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network_sim import NetworkSimulator


class FastSDNEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = 42, num_nodes: int = 14, num_links: int = 21,
                 link_capacity: float = 500.0, num_flows: int = 12,
                 max_episode_steps: int = 50, **sim_kwargs):
        super().__init__()
        self.sim = NetworkSimulator(
            num_nodes=num_nodes, num_links=num_links, seed=seed,
            link_capacity=link_capacity, num_flows=num_flows,
            max_episode_steps=max_episode_steps, **sim_kwargs)
        self.num_nodes = num_nodes
        self.num_links = num_links
        self.max_links = self.sim.max_links
        self.max_episode_steps = max_episode_steps

        obs_dim = self.num_nodes + 2 * self.max_links
        self.observation_space = spaces.Box(low=0.0, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=1.0, high=100.0,
                                       shape=(self.max_links,), dtype=np.float32)

        self.weights = np.full(self.max_links, 10.0, dtype=np.float32)
        self.last_metrics: dict = {}
        self.step_count = 0

    # ------------------------------------------------------------- utilities
    @property
    def edge_index(self) -> np.ndarray:
        """edge_index (2, max_links) สำหรับสร้าง GNN ใน policy"""
        return self.sim.edge_index_gnn

    def sample_flows(self, seed: Optional[int] = None) -> None:
        """บังคับ scenario เดียวกันให้ทุก method (ใช้ใน benchmark)"""
        self.sim.sample_flows(seed=seed)

    def set_weights(self, weights: np.ndarray) -> dict:
        """กำหนด weights แล้วคืน metrics ของรอบนั้น (ใช้กับ baseline Dijkstra)"""
        self.weights = np.clip(np.asarray(weights, dtype=np.float32), 1.0, 100.0)
        return self._refresh()

    def simulate_ecmp(self) -> dict:
        """Baseline ECMP: คืน metrics โดยไม่ต้องผ่าน step()"""
        self.last_metrics = self.sim.simulate_ecmp()
        return self.last_metrics

    def _refresh(self) -> dict:
        self.last_metrics = self.sim.simulate(self.weights)
        return self.last_metrics

    def _observation(self) -> np.ndarray:
        return self.sim.build_observation(self.weights, self.last_metrics["link_utilization"])

    # ------------------------------------------------------------ gymnasium
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        if seed is not None:
            self.sim.sample_flows(seed=seed)
        self.weights = np.full(self.max_links, 10.0, dtype=np.float32)  # เริ่มที่ OSPF-like
        self._refresh()
        return self._observation(), {}

    def step(self, action: np.ndarray):
        metrics = self.set_weights(action)
        reward = float(metrics["reward"])
        self.step_count += 1
        terminated = self.step_count >= self.max_episode_steps
        truncated = False
        info = {
            "throughput": metrics["throughput"],
            "latency": metrics["latency"],
            "packet_loss": metrics["packet_loss"],
            "mean_utilization": float(np.mean(metrics["link_utilization"])),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        m = self.last_metrics
        print(f"[FastSDNEnv] throughput={m.get('throughput', 0):.1f} Mbps | "
              f"latency={m.get('latency', 0):.2f} ms | "
              f"loss={m.get('packet_loss', 0):.4f} | reward={m.get('reward', 0):.6f}")

    def close(self):
        pass
