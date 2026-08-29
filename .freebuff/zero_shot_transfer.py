#!/usr/bin/env python3
"""
zero_shot_transfer.py — Zero-Shot Generalization Test
Train PPO+GNN on NSFNET → Evaluate on Abilene WITHOUT retraining

3 approaches compared:
  1. Zero-Shot (direct transfer): Load NSFNET model, pad obs, evaluate on Abilene
  2. Fine-Tune Transfer: Load NSFNET GNN encoder, retrain action head 5k steps on Abilene
  3. From Scratch: Train PPO+GNN from scratch on Abilene 5k steps (baseline)

All evaluated on same 100 seeds for fair comparison.
"""
import json, time, sys, os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from gymnasium import spaces
import gymnasium as gym

sys.path.insert(0, '/home/ino')
from network_sim import NetworkSimulator
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

# Import existing code
from train_gnn_v2 import (
    ShapedFastSDNEnv, SDNGraphFeatureExtractor,
    evaluate, evaluate_ospp, evaluate_optimal,
    NSFNET_EDGE_INDEX, ABILENE_EDGE_INDEX,
    NSFNET_NUM_NODES, NSFNET_NUM_EDGES,
    ABILENE_NUM_NODES, ABILENE_NUM_EDGES,
)


# ═══════════════════════════════════════════════════════════════
# Zero-Shot Wrapper: pad observations + truncate actions
# ═══════════════════════════════════════════════════════════════

class ZeroShotWrapper(gym.ObservationWrapper):
    """Wraps Abilene env to produce NSFNET-compatible observations (77 dims).
    Abilene: 12 nodes + 3*15 edges = 57 dims → pad to 77 dims.
    Action: model outputs 21 weights → use first 15 for Abilene links.
    """
    def __init__(self, env):
        super().__init__(env)
        self.nsfnet_obs_dim = NSFNET_NUM_NODES + 3 * NSFNET_NUM_EDGES  # 77
        self.ab_nodes = env.num_nodes   # 12
        self.ab_links = env.num_links   # 15
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(self.nsfnet_obs_dim,), dtype=np.float32
        )

    def observation(self, obs):
        """Pad Abilene obs (57) to NSFNET shape (77)."""
        padded = np.zeros(self.nsfnet_obs_dim, dtype=np.float32)
        # Copy node features (first 12 values)
        padded[:self.ab_nodes] = obs[:self.ab_nodes]
        # Copy edge features (15 edges × 3 = 45 values) starting at position 14
        edge_start_nsfnet = NSFNET_NUM_NODES  # 14
        edge_start_abilene = self.ab_nodes     # 12
        edge_size = self.ab_links * 3          # 45
        padded[edge_start_nsfnet:edge_start_nsfnet + edge_size] = \
            obs[edge_start_abilene:edge_start_abilene + edge_size]
        return padded

    def step(self, action):
        """Truncate 21-dim action to 15-dim for Abilene."""
        abilene_action = action[:self.ab_links].astype(np.float32)
        obs, reward, done, trunc, info = self.env.step(abilene_action)
        return self.observation(obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ═══════════════════════════════════════════════════════════════
# GNN with Abilene edge index (for fine-tune + from-scratch)
# ═══════════════════════════════════════════════════════════════

class AbileneGNNFeatureExtractor(BaseFeaturesExtractor):
    """GAT feature extractor adapted for Abilene topology."""
    def __init__(self, observation_space, num_nodes=12, num_edges=15,
                 hidden_dim=128, num_heads=4):
        super().__init__(observation_space, features_dim=128)
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.node_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations.shape[0]
        device = observations.device
        all_features = []
        for i in range(batch_size):
            obs = observations[i]
            node_feat = obs[:self.num_nodes].unsqueeze(1)
            node_emb = self.node_encoder(node_feat)
            edge_index = ABILENE_EDGE_INDEX.to(device)
            x = torch.relu(self.gat1(node_emb, edge_index))
            x = torch.relu(self.gat2(x, edge_index))
            x = torch.relu(self.gat3(x, edge_index))
            graph_emb = x.mean(dim=0)
            edge_start = self.num_nodes
            edge_data = obs[edge_start:edge_start + self.num_edges * 3]
            edge_attr = edge_data.view(self.num_edges, 3)
            edge_emb = self.edge_encoder(edge_attr).mean(dim=0)
            combined = torch.cat([graph_emb, edge_emb])
            features = self.output(combined)
            all_features.append(features)
        return torch.stack(all_features)


# ═══════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════

def eval_model(model, env, seeds=100, action_truncate=None):
    """Evaluate model on env for N seeds."""
    tputs, lats, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        action, _ = model.predict(obs, deterministic=True)
        if action_truncate is not None:
            action = action[:action_truncate]
        obs, reward, done, trunc, info = env.step(action)
        tputs.append(info['throughput'])
        lats.append(info['latency'])
        losses.append(info['packet_loss'] * 100)
    return {
        'throughput': float(np.mean(tputs)),
        'latency': float(np.mean(lats)),
        'loss': float(np.mean(losses)),
        'throughput_std': float(np.std(tputs)),
        'latency_std': float(np.std(lats)),
        'loss_std': float(np.std(losses)),
    }


def eval_ospf(env, seeds=100):
    """Evaluate OSPF baseline."""
    tputs, lats, losses = [], [], []
    weights = np.ones(env.max_links, dtype=np.float32) * 10.0
    for s in range(seeds):
        env.reset(seed=s)
        metrics = env.sim.simulate(weights)
        tputs.append(metrics['throughput'])
        lats.append(metrics['latency'])
        losses.append(metrics['packet_loss'] * 100)
    return {
        'throughput': float(np.mean(tputs)),
        'latency': float(np.mean(lats)),
        'loss': float(np.mean(losses)),
    }


def eval_optimal(env, seeds=100):
    """Evaluate Optimal (1/BW) baseline."""
    tputs, lats, losses = [], [], []
    caps = np.array(env.sim.link_capacities, dtype=np.float32)
    weights = np.clip(1000.0 / caps, 1.0, 100.0)
    for s in range(seeds):
        env.reset(seed=s)
        metrics = env.sim.simulate(weights)
        tputs.append(metrics['throughput'])
        lats.append(metrics['latency'])
        losses.append(metrics['packet_loss'] * 100)
    return {
        'throughput': float(np.mean(tputs)),
        'latency': float(np.mean(lats)),
        'loss': float(np.mean(losses)),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ZERO-SHOT TRANSFER TEST: NSFNET → Abilene")
    print("=" * 70)

    # ── 1. Baselines on Abilene ──
    print("\n[1/4] Evaluating baselines on Abilene...")
    ab_env = ShapedFastSDNEnv(topology='abilene', seed=42)
    print(f"  Abilene: {ab_env.num_nodes} nodes, {ab_env.num_links} links, "
          f"obs_dim={ab_env.observation_space.shape[0]}")

    ospf = eval_ospf(ab_env, seeds=100)
    optimal = eval_optimal(ab_env, seeds=100)
    print(f"  OSPF:    {ospf['throughput']:.2f} Mbps, {ospf['latency']:.2f}ms, {ospf['loss']:.2f}%")
    print(f"  Optimal: {optimal['throughput']:.2f} Mbps, {optimal['latency']:.2f}ms, {optimal['loss']:.2f}%")

    # ── 2. Zero-Shot Transfer (direct) ──
    print("\n[2/4] Zero-Shot Transfer: loading NSFNET model → Abilene...")
    model_path = "/home/ino/ppo_gnn_50k_best"
    if not os.path.exists(model_path + ".zip"):
        model_path = "/home/ino/ppo_gnn_nsfnet_50k"

    model_nsfnet = PPO.load(model_path)
    print(f"  Loaded: {model_path}.zip")

    # Wrap Abilene env for NSFNET-compatible obs
    wrapped_env = ZeroShotWrapper(ab_env)
    print(f"  Wrapped obs: {ab_env.observation_space.shape[0]} → {wrapped_env.observation_space.shape[0]} dims")

    # Evaluate with action truncation (21 → 15)
    zero_shot = eval_model(model_nsfnet, wrapped_env, seeds=100, action_truncate=15)
    gain = ((zero_shot['throughput'] - ospf['throughput']) / ospf['throughput']) * 100
    print(f"  Zero-Shot: {zero_shot['throughput']:.2f} Mbps ({gain:+.2f}% vs OSPF)")
    print(f"             {zero_shot['latency']:.2f}ms, {zero_shot['loss']:.2f}%")

    # ── 3. Fine-Tune Transfer (5k steps on Abilene) ──
    print("\n[3/4] Fine-Tune Transfer: NSFNET encoder + 5k steps on Abilene...")
    ft_env = ShapedFastSDNEnv(topology='abilene', seed=42)

    policy_kwargs_ft = dict(
        features_extractor_class=AbileneGNNFeatureExtractor,
        features_extractor_kwargs=dict(
            num_nodes=12, num_edges=15, hidden_dim=128, num_heads=4
        )
    )

    model_ft = PPO(
        "MlpPolicy", ft_env,
        learning_rate=5e-4, n_steps=256, batch_size=64, n_epochs=10,
        gamma=0.99, ent_coef=0.02, clip_range=0.2, verbose=0, seed=42,
        policy_kwargs=policy_kwargs_ft
    )

    # Transfer GNN weights from NSFNET model
    nsfnet_state = model_nsfnet.policy.features_extractor.state_dict()
    ab_state = model_ft.policy.features_extractor.state_dict()
    transferred = 0
    for key in ab_state:
        if key in nsfnet_state and ab_state[key].shape == nsfnet_state[key].shape:
            ab_state[key] = nsfnet_state[key]
            transferred += 1
    model_ft.policy.features_extractor.load_state_dict(ab_state)
    print(f"  Transferred {transferred} layers from NSFNET → Abilene GNN")

    # Fine-tune 5k steps
    t0 = time.time()
    model_ft.learn(total_timesteps=5000)
    ft_time = time.time() - t0
    print(f"  Fine-tuned in {ft_time:.1f}s ({ft_time/60:.1f} min)")

    fine_tune = eval_model(model_ft, ft_env, seeds=100)
    ft_gain = ((fine_tune['throughput'] - ospf['throughput']) / ospf['throughput']) * 100
    print(f"  Fine-Tune: {fine_tune['throughput']:.2f} Mbps ({ft_gain:+.2f}% vs OSPF)")
    print(f"             {fine_tune['latency']:.2f}ms, {fine_tune['loss']:.2f}%")

    # ── 4. From Scratch (5k steps on Abilene) ──
    print("\n[4/4] From Scratch: train PPO+GNN on Abilene 5k steps...")
    scratch_env = ShapedFastSDNEnv(topology='abilene', seed=42)

    policy_kwargs_scratch = dict(
        features_extractor_class=AbileneGNNFeatureExtractor,
        features_extractor_kwargs=dict(
            num_nodes=12, num_edges=15, hidden_dim=128, num_heads=4
        )
    )

    model_scratch = PPO(
        "MlpPolicy", scratch_env,
        learning_rate=5e-4, n_steps=256, batch_size=64, n_epochs=10,
        gamma=0.99, ent_coef=0.02, clip_range=0.2, verbose=0, seed=42,
        policy_kwargs=policy_kwargs_scratch
    )

    t0 = time.time()
    model_scratch.learn(total_timesteps=5000)
    scratch_time = time.time() - t0
    print(f"  Trained in {scratch_time:.1f}s ({scratch_time/60:.1f} min)")

    from_scratch = eval_model(model_scratch, scratch_env, seeds=100)
    sc_gain = ((from_scratch['throughput'] - ospf['throughput']) / ospf['throughput']) * 100
    print(f"  From Scratch: {from_scratch['throughput']:.2f} Mbps ({sc_gain:+.2f}% vs OSPF)")
    print(f"                {from_scratch['latency']:.2f}ms, {from_scratch['loss']:.2f}%")

    # ── 5. Summary ──
    print("\n" + "=" * 70)
    print("ZERO-SHOT TRANSFER RESULTS — Abilene Topology")
    print("=" * 70)
    results = {
        'OSPF': ospf,
        'Optimal (1/BW)': optimal,
        'Zero-Shot (NSFNET→Abilene)': zero_shot,
        'Fine-Tune 5k (NSFNET→Abilene)': fine_tune,
        'From Scratch 5k': from_scratch,
    }

    header = f"{'Method':<35} {'Throughput':>10} {'vs OSPF':>9} {'Latency':>10} {'Loss':>8}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        gain = ((r['throughput'] - ospf['throughput']) / ospf['throughput']) * 100
        print(f"{name:<35} {r['throughput']:>8.2f} {gain:>+8.2f}% "
              f"{r['latency']:>8.2f} {r['loss']:>6.2f}%")

    # Compute transfer efficiency
    scratch_only = from_scratch['throughput'] - ospf['throughput']
    ft_improve = fine_tune['throughput'] - ospf['throughput']
    zs_improve = zero_shot['throughput'] - ospf['throughput']

    print(f"\n--- Transfer Learning Efficiency ---")
    print(f"  From Scratch improvement: {scratch_only:+.2f} Mbps (100%)")
    print(f"  Fine-Tune improvement:    {ft_improve:+.2f} Mbps ({ft_improve/max(scratch_only,0.01)*100:.0f}% of scratch)")
    print(f"  Zero-Shot improvement:    {zs_improve:+.2f} Mbps ({zs_improve/max(scratch_only,0.01)*100:.0f}% of scratch)")
    print(f"  Fine-Tune time: {ft_time:.1f}s | Scratch time: {scratch_time:.1f}s | Speedup: {scratch_time/ft_time:.1f}x")

    # Save
    output = {
        'title': 'Zero-Shot Transfer: NSFNET → Abilene',
        'source_topology': 'NSFNET (14 nodes, 21 links)',
        'target_topology': 'Abilene (12 nodes, 15 links)',
        'nsfnet_model': model_path,
        'training_steps': 5000,
        'evaluation_seeds': 100,
        'results': results,
        'transfer_efficiency': {
            'zero_shot_improvement_pct': zs_improve,
            'finetune_improvement_pct': ft_improve,
            'scratch_improvement_pct': scratch_only,
            'finetune_vs_scratch': ft_improve / max(scratch_only, 0.01) * 100,
            'time_speedup': scratch_time / ft_time,
        }
    }

    outpath = '/home/ino/zero_shot_transfer_results.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")

    # Also save to local .freebuff
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zero_shot_transfer_results.json')
    with open(local_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {local_path}")


if __name__ == '__main__':
    main()
