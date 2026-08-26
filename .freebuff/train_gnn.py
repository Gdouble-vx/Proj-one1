#!/usr/bin/env python3
"""
Train PPO with GNN Feature Extractor on Asymmetric NSFNET Topology
Uses Graph Attention Network (GAT) to process network topology
"""
import json, time, sys, os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from gymnasium import spaces
import gymnasium as gym

sys.path.insert(0, '/home/ino')
from fast_sdn_env import FastSDNEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SDNGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    GNN Feature Extractor for SDN routing.
    Processes network topology (nodes, edges, edge features) using GAT.
    """
    
    def __init__(self, observation_space: spaces.Box, num_nodes: int = 14,
                 num_edges: int = 21, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__(observation_space, features_dim=128)
        
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_feat_dim = 1  # utilization
        self.edge_feat_dim = 3  # utilization, weight, bandwidth
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, hidden_dim),
            nn.ReLU()
        )
        
        # GAT layers for message passing
        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feat_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        
        all_features = []
        
        for i in range(batch_size):
            obs = observations[i]
            
            # Extract node features (first num_nodes values)
            node_feat = obs[:self.num_nodes].unsqueeze(1)  # (num_nodes, 1)
            node_emb = self.node_encoder(node_feat)  # (num_nodes, hidden)
            
            # GNN message passing
            # Create edge index (simple chain for now, will be overridden)
            edge_index = self._get_edge_index().to(device)
            edge_attr = obs[self.num_nodes:].view(-1, self.edge_feat_dim)  # (num_edges, 3)
            
            # GAT layers
            x = self.gat1(node_emb, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            x = torch.relu(x)
            
            # Global pooling
            graph_emb = x.mean(dim=0)  # (hidden,)
            
            # Edge features summary
            edge_emb = self.edge_encoder(edge_attr).mean(dim=0)  # (hidden,)
            
            # Concatenate
            combined = torch.cat([graph_emb, edge_emb])  # (hidden*2)
            features = self.output(combined)  # (128,)
            
            all_features.append(features)
        
        return torch.stack(all_features)
    
    def _get_edge_index(self):
        """Get edge index for the graph."""
        # Use actual topology edges from network_sim
        from network_sim import NetworkSimulator
        sim = NetworkSimulator(topology='nsfnet' if self.num_nodes == 14 else 'abilene')
        src = sim.edges_u.tolist()
        dst = sim.edges_v.tolist()
        # Make bidirectional
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        return edge_index


class SDNGNNPolicy(nn.Module):
    """GNN-based policy network for PPO."""
    
    def __init__(self, observation_space, action_space, num_nodes=14, num_edges=21):
        super().__init__()
        self.extractor = SDNGraphFeatureExtractor(
            observation_space, num_nodes, num_edges
        )
        self.policy_head = nn.Linear(128, action_space.shape[0])
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, obs):
        features = self.extractor(obs)
        return self.policy_head(features), self.value_head(features)


def make_gnn_env(topology='nsfnet', seed=42):
    """Create environment with GNN-compatible observation."""
    env = FastSDNEnv(topology=topology, seed=seed)
    
    # Ensure observation includes graph structure
    obs, _ = env.reset(seed=seed)
    print(f"Env obs shape: {obs.shape}")
    print(f"Num nodes: {env.sim.num_nodes}, Num edges: {env.sim.num_links}")
    
    return env


def evaluate(model, env, seeds=20):
    """Evaluate model."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        m = info if isinstance(info, dict) and 'throughput' in info else env.last_metrics
        throughputs.append(m['throughput'])
        latencies.append(m['latency'])
        losses.append(m['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }


def evaluate_ospp(env, seeds=20):
    """Evaluate OSPF baseline."""
    throughputs, latencies, losses = [], [], []
    ospf_weights = np.ones(env.max_links, dtype=np.float32)
    for s in range(seeds):
        env.reset(seed=s)
        m = env.set_weights(ospf_weights)
        throughputs.append(m['throughput'])
        latencies.append(m['latency'])
        losses.append(m['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }


def main():
    results = {}
    
    print("=" * 60)
    print("PPO + GNN (GAT) Training on Asymmetric NSFNET")
    print("=" * 60)
    
    # Create environment
    env = make_gnn_env(topology='nsfnet', seed=42)
    
    # OSPF baseline
    ospp = evaluate_ospp(env)
    print(f"\nOSPF Baseline: {ospp['throughput']:.1f} Mbps, {ospp['latency']:.1f}ms, {ospp['loss']:.1f}%")
    
    # Train at different timesteps
    for steps in [5000, 10000, 50000]:
        print(f"\n--- Training {steps} steps with GNN Policy ---")
        
        # Create policy with GNN feature extractor
        policy_kwargs = dict(
            features_extractor_class=SDNGraphFeatureExtractor,
            features_extractor_kwargs=dict(
                num_nodes=env.sim.num_nodes,
                num_edges=env.sim.num_links,
                hidden_dim=64,
                num_heads=4
            )
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            seed=42,
            policy_kwargs=policy_kwargs
        )
        
        t0 = time.time()
        model.learn(total_timesteps=steps)
        elapsed = time.time() - t0
        
        # Evaluate
        metrics = evaluate(model, env)
        gain = ((metrics['throughput'] - ospp['throughput']) / ospp['throughput']) * 100
        
        print(f"\nTraining: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Result: {metrics['throughput']:.1f} Mbps ({gain:+.1f}% vs OSPF)")
        print(f"Latency: {metrics['latency']:.1f}ms, Loss: {metrics['loss']:.1f}%")
        
        results[f'gnn_{steps}'] = {
            'throughput': metrics['throughput'],
            'latency': metrics['latency'],
            'loss': metrics['loss'],
            'improvement': gain,
            'training_time': elapsed
        }
        
        # Save model
        model.save(f"/home/ino/ppo_gnn_nsfnet_{steps}")
        print(f"Model saved: ppo_gnn_nsfnet_{steps}.zip")
    
    # Zero-shot transfer to Abilene
    print("\n" + "=" * 60)
    print("ZERO-SHOT TRANSFER: NSFNET -> Abilene")
    print("=" * 60)
    
    ab_env = make_gnn_env(topology='abilene', seed=42)
    ospp_ab = evaluate_ospp(ab_env)
    print(f"\nAbilene OSPF: {ospp_ab['throughput']:.1f} Mbps")
    
    for steps in [5000, 10000, 50000]:
        try:
            # Load model and create compatible policy
            model_path = f"/home/ino/ppo_gnn_nsfnet_{steps}"
            if os.path.exists(f"{model_path}.zip"):
                # Need to recreate env with correct obs space for loading
                model = PPO.load(model_path, env=ab_env)
                metrics = evaluate(model, ab_env)
                gain = ((metrics['throughput'] - ospp_ab['throughput']) / ospp_ab['throughput']) * 100
                print(f"Zero-shot {steps}: {metrics['throughput']:.1f} Mbps ({gain:+.1f}% vs OSPF)")
                results[f'gnn_{steps}_abilene'] = {
                    'throughput': metrics['throughput'],
                    'improvement': gain
                }
        except Exception as e:
            print(f"Zero-shot {steps} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"\n{'Method':<35} {'Tput':>8} {'Latency':>8} {'Loss':>8} {'vs OSPF':>8}")
    print("-" * 68)
    print(f"{'OSPF':<35} {ospp['throughput']:>7.1f}M {ospp['latency']:>7.1f}ms {ospp['loss']:>7.1f}% {'---':>8}")
    
    for steps in [5000, 10000, 50000]:
        m = results[f'gnn_{steps}']
        t = m['training_time']
        print(f"{'PPO+GNN '+str(steps)+'s ('+str(int(t))+'s)':<35} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}% {m['improvement']:>+7.1f}%")
    
    print(f"\n--- Zero-Shot Transfer (-> Abilene) ---")
    print(f"{'Abilene OSPF':<35} {ospp_ab['throughput']:>7.1f}M {'---':>8} {'---':>8} {'---':>8}")
    for steps in [5000, 10000, 50000]:
        key = f'gnn_{steps}_abilene'
        if key in results:
            m = results[key]
            print(f"{'Zero-shot '+str(steps)+'s':<35} {m['throughput']:>7.1f}M {'---':>8} {'---':>8} {m['improvement']:>+7.1f}%")
    
    # Save results
    with open("/home/ino/gnn_training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/ino/gnn_training_results.json")


if __name__ == "__main__":
    main()
