#!/usr/bin/env python3
"""Evaluate best GNN model vs OSPF vs Optimal — 100 seeds for statistical significance"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/home/ino')

from train_gnn_v2 import ShapedFastSDNEnv, SDNGraphFeatureExtractor, evaluate, evaluate_ospp, evaluate_optimal
from stable_baselines3 import PPO

def main():
    print("=" * 70)
    print("EVALUATION: PPO+GNN Best Model vs Baselines (100 seeds)")
    print("=" * 70)

    env = ShapedFastSDNEnv(topology='nsfnet', seed=42)

    # Baselines
    print("\nEvaluating OSPF (100 seeds)...")
    t0 = time.time()
    ospp = evaluate_ospp(env, seeds=100)
    print(f"  OSPF:    {ospp['throughput']:.2f} Mbps | {ospp['latency']:.2f}ms | {ospp['loss']:.2f}% ({time.time()-t0:.0f}s)")

    print("Evaluating Optimal 1/BW (100 seeds)...")
    t0 = time.time()
    optimal = evaluate_optimal(env, seeds=100)
    print(f"  Optimal: {optimal['throughput']:.2f} Mbps (+{((optimal['throughput']-ospp['throughput'])/ospp['throughput'])*100:.2f}%) ({time.time()-t0:.0f}s)")

    # Load best model
    print("\nEvaluating PPO+GNN Best (100 seeds)...")
    policy_kwargs = dict(
        features_extractor_class=SDNGraphFeatureExtractor,
        features_extractor_kwargs=dict(
            num_nodes=env.num_nodes, num_edges=env.num_links,
            hidden_dim=128, num_heads=4
        )
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
    model = PPO.load("/home/ino/ppo_gnn_best", env=env)

    t0 = time.time()
    metrics_best = evaluate(model, env, seeds=100)
    gain = ((metrics_best['throughput'] - ospp['throughput']) / ospp['throughput']) * 100
    print(f"  Best:    {metrics_best['throughput']:.2f} Mbps ({gain:+.2f}%) | {metrics_best['latency']:.2f}ms | {metrics_best['loss']:.2f}% ({time.time()-t0:.0f}s)")

    # Also evaluate 10k model
    print("\nEvaluating PPO+GNN 10k (100 seeds)...")
    model_10k = PPO.load("/home/ino/ppo_gnn_nsfnet_10000", env=env)
    t0 = time.time()
    metrics_10k = evaluate(model_10k, env, seeds=100)
    gain_10k = ((metrics_10k['throughput'] - ospp['throughput']) / ospp['throughput']) * 100
    print(f"  10k:     {metrics_10k['throughput']:.2f} Mbps ({gain_10k:+.2f}%) | {metrics_10k['latency']:.2f}ms | {metrics_10k['loss']:.2f}% ({time.time()-t0:.0f}s)")

    # Also evaluate the old 5k model (v1)
    print("\nEvaluating PPO+GNN 5k v1 (100 seeds)...")
    try:
        model_5k = PPO.load("/home/ino/ppo_gnn_nsfnet_5000", env=env)
        t0 = time.time()
        metrics_5k = evaluate(model_5k, env, seeds=100)
        gain_5k = ((metrics_5k['throughput'] - ospp['throughput']) / ospp['throughput']) * 100
        print(f"  5k v1:   {metrics_5k['throughput']:.2f} Mbps ({gain_5k:+.2f}%) | {metrics_5k['latency']:.2f}ms | {metrics_5k['loss']:.2f}% ({time.time()-t0:.0f}s)")
    except:
        print("  5k v1: load failed")
        metrics_5k = None

    # Summary
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (100 seeds)")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Tput':>10} {'vs OSPF':>10} {'Latency':>10} {'Loss':>10}")
    print("-" * 75)
    print(f"{'OSPF (hop-count)':<35} {ospp['throughput']:>9.2f}M {'---':>10} {ospp['latency']:>9.2f}ms {ospp['loss']:>9.2f}%")
    print(f"{'Optimal (1/BW)':<35} {optimal['throughput']:>9.2f}M {((optimal['throughput']-ospp['throughput'])/ospp['throughput'])*100:>+9.2f}% {optimal['latency']:>9.2f}ms {optimal['loss']:>9.2f}%")
    print(f"{'PPO+GNN Best (5k ckpt)':<35} {metrics_best['throughput']:>9.2f}M {gain:>+9.2f}% {metrics_best['latency']:>9.2f}ms {metrics_best['loss']:>9.2f}%")
    print(f"{'PPO+GNN 10k':<35} {metrics_10k['throughput']:>9.2f}M {gain_10k:>+9.2f}% {metrics_10k['latency']:>9.2f}ms {metrics_10k['loss']:>9.2f}%")
    if metrics_5k:
        gain_5k_v = ((metrics_5k['throughput'] - ospp['throughput']) / ospp['throughput']) * 100
        print(f"{'PPO+GNN 5k v1':<35} {metrics_5k['throughput']:>9.2f}M {gain_5k_v:>+9.2f}% {metrics_5k['latency']:>9.2f}ms {metrics_5k['loss']:>9.2f}%")

    # Save
    results = {
        'ospp': ospp,
        'optimal': optimal,
        'gnn_best': metrics_best,
        'gnn_best_improvement_pct': gain,
        'gnn_10k': metrics_10k,
        'gnn_10k_improvement_pct': gain_10k,
    }
    if metrics_5k:
        results['gnn_5k_v1'] = metrics_5k
        results['gnn_5k_v1_improvement_pct'] = ((metrics_5k['throughput'] - ospp['throughput']) / ospp['throughput']) * 100

    with open("/home/ino/gnn_v2_final_eval.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to /home/ino/gnn_v2_final_eval.json")


if __name__ == "__main__":
    main()
