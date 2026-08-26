#!/usr/bin/env python3
"""
PPO + GNN (GAT) Training v2 — Fixed edge index + reward shaping
Trains on Asymmetric NSFNET topology, evaluates vs OSPF, zero-shot to Abilene
"""
import json, time, sys, os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from gymnasium import spaces
import gymnasium as gym

sys.path.insert(0, '/home/ino')
from network_sim import NetworkSimulator
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

# ============================================================
# NSFNET Topology: Hardcoded from network_sim.py
# ============================================================
NSFNET_EDGES_U = [0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11]
NSFNET_EDGES_V = [1, 2, 7, 2, 6, 3, 4, 5, 5, 6, 12, 13, 7, 8, 9, 11, 10, 12, 11, 13, 12]
NSFNET_CAPS = [150, 80, 100, 20, 15, 15, 100, 200, 150, 150, 15, 80, 100, 20, 150, 200, 100, 100, 200, 15, 80]
NSFNET_NUM_NODES = 14
NSFNET_NUM_EDGES = 21

# Bidirectional edge index
src = NSFNET_EDGES_U
dst = NSFNET_EDGES_V
NSFNET_EDGE_INDEX = torch.tensor([src + dst, dst + src], dtype=torch.long)  # (2, 42)

# Abilene topology (for zero-shot)
ABILENE_EDGES_U = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10]
ABILENE_EDGES_V = [1, 4, 2, 3, 5, 4, 6, 6, 5, 7, 10, 10, 9, 10, 11]
ABILENE_NUM_NODES = 12
ABILENE_NUM_EDGES = 15
src_a = ABILENE_EDGES_U
dst_a = ABILENE_EDGES_V
ABILENE_EDGE_INDEX = torch.tensor([src_a + dst_a, dst_a + src_a], dtype=torch.long)


# ============================================================
# GNN Feature Extractor
# ============================================================
class SDNGraphFeatureExtractor(BaseFeaturesExtractor):
    """GAT-based feature extractor for SDN routing graph."""

    def __init__(self, observation_space: spaces.Box, num_nodes: int = 14,
                 num_edges: int = 21, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__(observation_space, features_dim=128)

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_feat_dim = 1   # utilization per node
        self.edge_feat_dim = 3   # utilization, weight, bandwidth per edge

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # GAT layers
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Combine node + edge embeddings
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        all_features = []

        for i in range(batch_size):
            obs = observations[i]

            # Node features: first num_nodes values (utilization)
            node_feat = obs[:self.num_nodes].unsqueeze(1)  # (num_nodes, 1)
            node_emb = self.node_encoder(node_feat)  # (num_nodes, hidden)

            # GNN message passing with correct edge index
            edge_index = NSFNET_EDGE_INDEX.to(device)
            x = self.gat1(node_emb, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            x = torch.relu(x)
            x = self.gat3(x, edge_index)
            x = torch.relu(x)

            # Global pooling
            graph_emb = x.mean(dim=0)  # (hidden,)

            # Edge features: after node features
            edge_start = self.num_nodes
            edge_data = obs[edge_start:edge_start + self.num_edges * self.edge_feat_dim]
            edge_attr = edge_data.view(self.num_edges, self.edge_feat_dim)  # (num_edges, 3)
            edge_emb = self.edge_encoder(edge_attr).mean(dim=0)  # (hidden,)

            # Combine
            combined = torch.cat([graph_emb, edge_emb])
            features = self.output(combined)  # (128,)

            all_features.append(features)

        return torch.stack(all_features)


# ============================================================
# Custom FastSDNEnv with reward shaping
# ============================================================
class ShapedFastSDNEnv(gym.Env):
    """FastSDNEnv with reward shaping that penalizes narrow link congestion."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, topology='nsfnet', seed=42):
        super().__init__()
        self.sim = NetworkSimulator(topology=topology, seed=seed)
        self.num_nodes = self.sim.num_nodes
        self.num_links = self.sim.num_links
        self.max_links = self.sim.max_links

        obs_dim = self.num_nodes + 3 * self.max_links
        self.observation_space = spaces.Box(low=0.0, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=1.0, high=100.0,
                                       shape=(self.max_links,), dtype=np.float32)

        self.weights = np.full(self.max_links, 10.0, dtype=np.float32)
        self.last_metrics = {}
        self.step_count = 0
        self.last_throughput = 0.0
        self.seed_val = seed

        # Store link capacities for reward shaping
        self.link_caps = np.array(self.sim.link_capacities, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        if seed is not None:
            self.seed_val = seed
        self.sim.sample_flows(seed=self.seed_val)
        self.weights = np.full(self.max_links, 10.0, dtype=np.float32)
        obs = self.sim.build_observation(self.weights,
                                         self.sim.simulate(self.weights)["link_utilization"])
        self.last_throughput = 0.0
        return obs, {}

    def step(self, action):
        # Clip action to valid range
        self.weights = np.clip(action.astype(np.float32), 1.0, 100.0)

        # Simulate with new weights
        metrics = self.sim.simulate(self.weights)
        self.last_metrics = metrics
        self.step_count += 1

        # ---- REWARD SHAPING ----
        throughput = metrics["throughput"]
        latency = metrics["latency"]
        loss = metrics["packet_loss"]
        link_util = np.array(metrics["link_utilization"])

        # Base reward: throughput normalized
        reward = throughput / 100.0  # scale down

        # Penalty for high utilization on narrow links (< 50 Mbps capacity)
        narrow_mask = self.link_caps < 50.0
        narrow_util = link_util[narrow_mask]
        if len(narrow_util) > 0:
            avg_narrow_util = np.mean(narrow_util)
            # Heavy penalty when narrow links are congested
            if avg_narrow_util > 0.8:
                reward -= (avg_narrow_util - 0.8) * 50.0
            # Bonus for keeping narrow links underutilized
            if avg_narrow_util < 0.3:
                reward += (0.3 - avg_narrow_util) * 10.0

        # Penalty for packet loss
        reward -= loss * 100.0

        # Bonus for load balancing (low std of utilization)
        util_std = np.std(link_util)
        reward -= util_std * 5.0  # penalize uneven distribution

        # Small exploration bonus: reward changing weights from uniform
        weight_diff = np.std(self.weights)
        if weight_diff > 5.0:
            reward += 1.0  # bonus for non-uniform weights

        # Penalize high latency
        if latency > 50:
            reward -= (latency - 50) / 10.0

        # Done after 1 step (each step is a full simulation)
        done = True
        truncated = False
        info = {
            "throughput": throughput,
            "latency": latency,
            "packet_loss": loss,
            "link_utilization": link_util,
        }

        obs = self.sim.build_observation(self.weights, link_util)
        return obs, reward, done, truncated, info

    @property
    def edge_index(self):
        return self.sim.edge_index_gnn


# ============================================================
# OSPF Baseline
# ============================================================
def evaluate_ospp(env, seeds=50):
    """Evaluate OSPF baseline (uniform weights)."""
    throughputs, latencies, losses = [], [], []
    ospf_weights = np.ones(env.max_links, dtype=np.float32) * 10.0
    for s in range(seeds):
        env.reset(seed=s)
        env.weights = ospf_weights.copy()
        metrics = env.sim.simulate(ospf_weights)
        throughputs.append(metrics['throughput'])
        latencies.append(metrics['latency'])
        losses.append(metrics['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }


def evaluate_optimal(env, seeds=50):
    """Evaluate optimal (1/BW) baseline."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        env.reset(seed=s)
        # Weight = inverse of bandwidth (prefer high-BW links)
        caps = np.array(env.sim.link_capacities, dtype=np.float32)
        optimal_weights = 1000.0 / caps
        optimal_weights = np.clip(optimal_weights, 1.0, 100.0)
        env.weights = optimal_weights.copy()
        metrics = env.sim.simulate(optimal_weights)
        throughputs.append(metrics['throughput'])
        latencies.append(metrics['latency'])
        losses.append(metrics['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }


# ============================================================
# Training Callback
# ============================================================
class TrainingMonitor(BaseCallback):
    def __init__(self, eval_env, ospp_metrics, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.ospp = ospp_metrics
        self.best_throughput = 0

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            metrics = evaluate(self.model, self.eval_env, seeds=10)
            gain = ((metrics['throughput'] - self.ospp['throughput']) / self.ospp['throughput']) * 100
            if metrics['throughput'] > self.best_throughput:
                self.best_throughput = metrics['throughput']
                self.model.save(f"/home/ino/ppo_gnn_best")
            print(f"  [Step {self.n_calls}] Tput={metrics['throughput']:.1f}Mbps ({gain:+.1f}% vs OSPF) "
                  f"Lat={metrics['latency']:.1f}ms Loss={metrics['loss']:.1f}%")
        return True


def evaluate(model, env, seeds=20):
    """Evaluate model on multiple seeds."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        throughputs.append(info['throughput'])
        latencies.append(info['latency'])
        losses.append(info['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }


# ============================================================
# Main
# ============================================================
def main():
    results = {}

    print("=" * 70)
    print("PPO + GNN (GAT v2) — Fixed Edge Index + Reward Shaping")
    print("Topology: Asymmetric NSFNET (14 nodes, 21 links)")
    print("=" * 70)

    # Create shaped environment
    env = ShapedFastSDNEnv(topology='nsfnet', seed=42)
    eval_env = ShapedFastSDNEnv(topology='nsfnet', seed=99)

    print(f"Obs dim: {env.observation_space.shape[0]} (nodes={env.num_nodes}, edges*3={env.num_links*3})")
    print(f"Action dim: {env.action_space.shape[0]}")
    print(f"Edge index shape: {NSFNET_EDGE_INDEX.shape}")

    # Baselines
    ospp = evaluate_ospp(env, seeds=50)
    optimal = evaluate_optimal(env, seeds=50)
    print(f"\n{'='*70}")
    print(f"BASELINES (50 seeds each)")
    print(f"{'='*70}")
    print(f"OSPF:    {ospp['throughput']:.1f} Mbps | {ospp['latency']:.1f}ms | {ospp['loss']:.1f}% loss")
    print(f"Optimal: {optimal['throughput']:.1f} Mbps (+{((optimal['throughput']-ospp['throughput'])/ospp['throughput'])*100:.1f}%)")

    results['ospp'] = ospp
    results['optimal'] = optimal

    # Train at different steps
    for steps in [10000, 30000, 50000]:
        print(f"\n{'='*70}")
        print(f"TRAINING PPO+GNN: {steps} steps")
        print(f"{'='*70}")

        policy_kwargs = dict(
            features_extractor_class=SDNGraphFeatureExtractor,
            features_extractor_kwargs=dict(
                num_nodes=env.num_nodes,
                num_edges=env.num_links,
                hidden_dim=128,
                num_heads=4
            )
        )

        callback = TrainingMonitor(eval_env=eval_env, ospp_metrics=ospp)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.02,  # encourage exploration
            clip_range=0.2,
            verbose=0,
            seed=42,
            policy_kwargs=policy_kwargs
        )

        t0 = time.time()
        model.learn(total_timesteps=steps, callback=callback)
        elapsed = time.time() - t0

        # Final evaluation with 50 seeds
        metrics = evaluate(model, eval_env, seeds=50)
        gain = ((metrics['throughput'] - ospp['throughput']) / ospp['throughput']) * 100

        print(f"\nRESULT ({elapsed:.0f}s = {elapsed/60:.1f}min):")
        print(f"  Throughput: {metrics['throughput']:.1f} Mbps ({gain:+.1f}% vs OSPF)")
        print(f"  Latency:    {metrics['latency']:.1f}ms")
        print(f"  Loss:       {metrics['loss']:.1f}%")

        results[f'gnn_{steps}'] = {
            'throughput': metrics['throughput'],
            'latency': metrics['latency'],
            'loss': metrics['loss'],
            'improvement_pct': gain,
            'training_time_s': elapsed,
        }

        model.save(f"/home/ino/ppo_gnn_nsfnet_{steps}")
        print(f"  Saved: ppo_gnn_nsfnet_{steps}.zip")

    # ============================================================
    # Zero-Shot Transfer to Abilene
    # ============================================================
    print(f"\n{'='*70}")
    print("ZERO-SHOT TRANSFER: NSFNET → Abilene")
    print(f"{'='*70}")

    ab_env = ShapedFastSDNEnv(topology='abilene', seed=42)
    ospp_ab = evaluate_ospp(ab_env, seeds=50)
    print(f"Abilene OSPF: {ospp_ab['throughput']:.1f} Mbps")

    results['abilene_ospp'] = ospp_ab

    for steps in [10000, 30000, 50000]:
        model_path = f"/home/ino/ppo_gnn_nsfnet_{steps}"
        if os.path.exists(f"{model_path}.zip"):
            try:
                # Need to re-create model with correct obs shape for Abilene
                ab_env_temp = ShapedFastSDNEnv(topology='abilene', seed=42)
                policy_kwargs_ab = dict(
                    features_extractor_class=SDNGraphFeatureExtractor,
                    features_extractor_kwargs=dict(
                        num_nodes=ab_env_temp.num_nodes,
                        num_edges=ab_env_temp.num_links,
                        hidden_dim=128,
                        num_heads=4
                    )
                )
                model = PPO("MlpPolicy", ab_env_temp, policy_kwargs=policy_kwargs_ab)
                model = PPO.load(model_path, env=ab_env_temp)

                metrics = evaluate(model, ab_env_temp, seeds=50)
                gain = ((metrics['throughput'] - ospp_ab['throughput']) / ospp_ab['throughput']) * 100
                print(f"Zero-shot {steps}: {metrics['throughput']:.1f} Mbps ({gain:+.1f}% vs OSPF)")
                results[f'gnn_{steps}_abilene'] = {
                    'throughput': metrics['throughput'],
                    'latency': metrics['latency'],
                    'loss': metrics['loss'],
                    'improvement_pct': gain,
                }
            except Exception as e:
                print(f"Zero-shot {steps} failed: {e}")

    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Method':<40} {'Tput':>8} {'vs OSPF':>8} {'Latency':>8} {'Loss':>8}")
    print("-" * 72)
    print(f"{'OSPF (hop-count)':<40} {ospp['throughput']:>7.1f}M {'---':>8} {ospp['latency']:>7.1f}ms {ospp['loss']:>7.1f}%")
    print(f"{'Optimal (1/BW)':<40} {optimal['throughput']:>7.1f}M {((optimal['throughput']-ospp['throughput'])/ospp['throughput'])*100:>+7.1f}% {optimal['latency']:>7.1f}ms {optimal['loss']:>7.1f}%")

    for steps in [10000, 30000, 50000]:
        m = results.get(f'gnn_{steps}')
        if m:
            print(f"{'PPO+GNN '+str(steps//1000)+'k ('+str(int(m['training_time_s']))+'s)':<40} {m['throughput']:>7.1f}M {m['improvement_pct']:>+7.1f}% {m['latency']:>7.1f}ms {m['loss']:>7.1f}%")

    print(f"\n--- Zero-Shot Transfer (→ Abilene) ---")
    print(f"{'Abilene OSPF':<40} {ospp_ab['throughput']:>7.1f}M {'---':>8}")
    for steps in [10000, 30000, 50000]:
        m = results.get(f'gnn_{steps}_abilene')
        if m:
            print(f"{'Zero-shot '+str(steps//1000)+'k':<40} {m['throughput']:>7.1f}M {m['improvement_pct']:>+7.1f}%")

    # Save results
    with open("/home/ino/gnn_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/ino/gnn_v2_results.json")


if __name__ == "__main__":
    main()
