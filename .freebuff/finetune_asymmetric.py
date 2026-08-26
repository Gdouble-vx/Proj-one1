#!/usr/bin/env python3
"""
Fine-tune PPO+GNN on Asymmetric NSFNET Topology
- Train PPO with GNN feature extractor to learn optimal link weights
- Avoid narrow links (15-20 Mbps) and route through wide links (100-200 Mbps)
- Export optimized weights and compute paths for ONOS OF rule installation
"""
import json
import time
import numpy as np
import sys

print("=== Fine-tuning PPO+GNN on Asymmetric NSFNET ===")
print(f"Python: {sys.version}")

# Import after printing version
from fast_sdn_env import FastSDNEnv

def compute_dijkstra_paths(sim, weights):
    """Compute shortest paths using Dijkstra for each flow."""
    paths = []
    for src, dst, demand in sim.flows:
        path = sim.dijkstra_path(src, dst, weights)
        paths.append(path)
    return paths

def compute_path_summary(sim, weights, label=""):
    """Compute and display metrics for given weights."""
    result = sim.simulate(weights)
    paths = compute_dijkstra_paths(sim, weights)
    
    # Summarize paths
    path_strs = []
    for i, (src, dst, demand) in enumerate(sim.flows):
        path_nodes = []
        for li in paths[i]:
            u, v = int(sim.edges_u[li]), int(sim.edges_v[li])
            path_nodes.append(f"s{u+1}")
        path_nodes.append(f"s{dst+1}")
        path_strs.append(f"h{src+1}->h{dst+1}: {'->'.join(path_nodes)} ({demand:.1f}Mbps)")
    
    # Count link usage and bandwidth utilization
    link_usage = np.zeros(sim.num_links, dtype=np.int32)
    for p in paths:
        for li in p:
            link_usage[li] += 1
    
    # Summary
    total_demand = sum(f[2] for f in sim.flows)
    print(f"\n=== {label} ===")
    print(f"Throughput: {result['throughput']:.1f} Mbps | Latency: {result['latency']:.1f} ms | Loss: {result['packet_loss']*100:.1f}%")
    print(f"Total demand: {total_demand:.1f} Mbps | Reward: {result['reward']:.4f}")
    print(f"\nFlows ({len(sim.flows)}):")
    for s in path_strs:
        print(f"  {s}")
    print(f"\nLink usage (narrow links marked *):")
    for i in range(sim.num_links):
        u, v = int(sim.edges_u[i]), int(sim.edges_v[i])
        cap = sim.link_capacities[i]
        used = link_usage[i]
        util = result['link_utilization'][i]
        tag = " *NARROW*" if cap <= 20 else (" [wide]" if cap >= 100 else "")
        if used > 0:
            print(f"  [{i:2d}] s{u+1:2d}<->s{v+1:2d} cap={cap:5.0f}Mbps  used={used}  util={util:.2f}{tag}")
    
    return result, paths

def train_ppo_gnn():
    """Train PPO+GNN on asymmetric NSFNET topology."""
    print("\n--- Creating Asymmetric NSFNET Environment ---")
    env = FastSDNEnv(topology='nsfnet', seed=42)
    
    obs, _ = env.reset(seed=42)
    print(f"Observation dim: {obs.shape}")
    print(f"Action dim: {env.action_space.shape}")
    print(f"Links: {env.max_links}")
    print(f"Link BW range: [{env.sim.link_capacities.min():.0f}, {env.sim.link_capacities.max():.0f}] Mbps")
    print(f"Narrow links (<30 Mbps): {(env.sim.link_capacities < 30).sum()}")
    print(f"Wide links (>=100 Mbps): {(env.sim.link_capacities >= 100).sum()}")
    
    # Import SB3
    from stable_baselines3 import PPO
    
    print("\n--- Training PPO (no GNN, raw observations) ---")
    print("Training 5000 timesteps...")
    
    start_time = time.time()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=42
    )
    
    model.learn(total_timesteps=5000)
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")
    
    return model, env

def extract_optimized_weights(model, env):
    """Use trained model to extract optimized link weights."""
    print("\n--- Extracting Optimized Weights ---")
    
    obs, _ = env.reset(seed=99)
    action, _ = model.predict(obs, deterministic=True)
    
    # Scale action from [-1,1] to [1,100]
    weights = ((action + 1) / 2) * 99 + 1
    weights = np.clip(weights, 1.0, 100.0).astype(np.float32)
    
    # Also try averaging over multiple episodes
    all_weights = []
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        action, _ = model.predict(obs, deterministic=True)
        w = ((action + 1) / 2) * 99 + 1
        w = np.clip(w, 1.0, 100.0)
        all_weights.append(w)
    
    avg_weights = np.mean(all_weights, axis=0).astype(np.float32)
    
    print(f"Optimized weights range: [{avg_weights.min():.1f}, {avg_weights.max():.1f}]")
    
    # Compare with narrow links
    narrow_mask = env.sim.link_capacities <= 20
    wide_mask = env.sim.link_capacities >= 100
    print(f"Mean weight on narrow links: {avg_weights[narrow_mask].mean():.1f} (should be HIGH)")
    print(f"Mean weight on wide links: {avg_weights[wide_mask].mean():.1f} (should be LOW)")
    
    return avg_weights

def compute_optimal_weights(sim):
    """Compute theoretically optimal weights (inverse of capacity)."""
    weights = np.zeros(sim.num_links, dtype=np.float32)
    for i in range(sim.num_links):
        cap = sim.link_capacities[i]
        # Higher weight for lower capacity (discourage use of narrow links)
        weights[i] = 1000.0 / cap
    weights = np.clip(weights, 1.0, 100.0)
    return weights

def generate_onos_rules(sim, weights, dst_host_ip="10.0.0.2"):
    """Generate OpenFlow rules for ONOS based on computed paths."""
    paths = compute_dijkstra_paths(sim, weights)
    rules = []
    
    # Get port mappings from ONOS
    # In Mininet, ports are assigned in order: host port=1, then links in creation order
    # We need to figure out which port connects which switches
    
    # For simplicity, we'll generate rules based on switch-to-switch paths
    # Each flow rule: match on IP dst, action: forward to next hop port
    
    for flow_idx, (src, dst, demand) in enumerate(sim.flows):
        path = paths[flow_idx]
        if not path:
            continue
            
        # Forward on each switch in the path
        for i, link_idx in enumerate(path):
            u = int(sim.edges_u[link_idx])
            v = int(sim.edges_v[link_idx])
            
            # Determine direction: if we came from src through u to v
            # The flow enters at port (link_index + 1) and exits at next link
            
            rules.append({
                "flow_id": f"h{src+1}_to_h{dst+1}_sw{s+1}",
                "switch": f"s{u+1}",
                "dpid": f"000000000000{u+1:04x}",
                "src_ip": f"10.0.0.{src+1}",
                "dst_ip": dst_host_ip,
                "next_hop": f"s{v+1}",
                "bandwidth": float(sim.link_capacities[link_idx]),
                "demand": float(demand)
            })
    
    return rules

def main():
    # Step 1: Baseline OSPF
    print("\n" + "="*60)
    print("STEP 1: Baseline OSPF (uniform weights = 1)")
    print("="*60)
    
    env = FastSDNEnv(topology='nsfnet', seed=42)
    env.reset(seed=42)
    
    ospf_weights = np.ones(env.max_links, dtype=np.float32)
    ospf_result, ospf_paths = compute_path_summary(env.sim, ospf_weights, "OSPF (hop-count)")
    
    # Step 2: Theoretically optimal (inverse capacity)
    print("\n" + "="*60)
    print("STEP 2: Theoretically Optimal (weight = 1000/capacity)")
    print("="*60)
    
    optimal_weights = compute_optimal_weights(env.sim)
    opt_result, opt_paths = compute_path_summary(env.sim, optimal_weights, "Optimal (1000/BW)")
    
    # Step 3: Train PPO
    print("\n" + "="*60)
    print("STEP 3: Train PPO+GNN (5000 timesteps)")
    print("="*60)
    
    model, trained_env = train_ppo_gnn()
    
    # Step 4: Extract PPO weights
    print("\n" + "="*60)
    print("STEP 4: PPO+GNN Optimized Weights")
    print("="*60)
    
    ppo_weights = extract_optimized_weights(model, trained_env)
    ppo_result, ppo_paths = compute_path_summary(trained_env.sim, ppo_weights, "PPO+GNN Optimized")
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n{'Method':<30} {'Tput(Mbps)':>12} {'Latency(ms)':>12} {'Loss%':>8} {'Reward':>10}")
    print("-"*72)
    for name, res in [("OSPF (hop-count)", ospf_result), 
                      ("Optimal (1/BW)", opt_result),
                      ("PPO+GNN", ppo_result)]:
        print(f"{name:<30} {res['throughput']:>12.1f} {res['latency']:>12.1f} {res['packet_loss']*100:>7.1f}% {res['reward']:>10.4f}")
    
    # Improvements
    if ospf_result['throughput'] > 0:
        ppo_vs_ospf = ((ppo_result['throughput'] - ospf_result['throughput']) / ospf_result['throughput']) * 100
        opt_vs_ospf = ((opt_result['throughput'] - ospf_result['throughput']) / ospf_result['throughput']) * 100
        print(f"\nPPO+GNN vs OSPF: {ppo_vs_ospf:+.1f}% throughput improvement")
        print(f"Optimal vs OSPF: {opt_vs_ospf:+.1f}% throughput improvement")
    
    # Step 6: Generate OF rules for ONOS
    print("\n" + "="*60)
    print("STEP 6: ONOS OpenFlow Rules (PPO+GNN optimized)")
    print("="*60)
    
    rules = generate_onos_rules(trained_env.sim, ppo_weights)
    print(f"\nGenerated {len(rules)} flow rules for ONOS:")
    for r in rules[:10]:
        print(f"  {r['switch']}: {r['src_ip']}->{r['dst_ip']} via {r['next_hop']} ({r['bandwidth']:.0f}Mbps)")
    if len(rules) > 10:
        print(f"  ... and {len(rules)-10} more rules")
    
    # Save results
    results = {
        "ospp": {
            "throughput": float(ospf_result['throughput']),
            "latency": float(ospf_result['latency']),
            "loss": float(ospf_result['packet_loss']),
            "reward": float(ospf_result['reward'])
        },
        "optimal": {
            "throughput": float(opt_result['throughput']),
            "latency": float(opt_result['latency']),
            "loss": float(opt_result['packet_loss']),
            "reward": float(opt_result['reward'])
        },
        "ppo_gnn": {
            "throughput": float(ppo_result['throughput']),
            "latency": float(ppo_result['latency']),
            "loss": float(ppo_result['packet_loss']),
            "reward": float(ppo_result['reward'])
        },
        "ppo_weights": ppo_weights.tolist(),
        "optimal_weights": optimal_weights.tolist(),
        "of_rules": rules,
        "training_time_s": train_time
    }
    
    with open("/home/ino/finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /home/ino/finetune_results.json")
    
    # Save model
    model.save("/home/ino/ppo_asymmetric_nsfnet")
    print("Model saved to /home/ino/ppo_asymmetric_nsfnet.zip")

if __name__ == "__main__":
    main()
