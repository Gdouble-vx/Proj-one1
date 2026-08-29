#!/usr/bin/env python3
"""Run Experiments 3-5 (transfer learning) using the best NSFNET model."""
import json, time, sys, os
import numpy as np

sys.path.insert(0, '/home/ino')
from improved_transfer_learning import (
    ImprovedSDNEnv, ImprovedGNNExtractor, MAX_NODES, MAX_LINKS,
    evaluate, evaluate_ospp, evaluate_optimal, freeze_gnn_layers, unfreeze_all
)
from stable_baselines3 import PPO

def main():
    print("=" * 70)
    print("EXPERIMENTS 3-5: Transfer Learning on Abilene")
    print("=" * 70)

    ab_env = ImprovedSDNEnv(topology='abilene', seed=42)
    ab_ospp = evaluate_ospp(ab_env, seeds=50)
    ab_optimal = evaluate_optimal(ab_env, seeds=50)

    print(f"Abilene OSPF: {ab_ospp['throughput']:.1f} Mbps")
    print(f"Abilene Optimal: {ab_optimal['throughput']:.1f} Mbps")

    policy_kwargs = dict(
        features_extractor_class=ImprovedGNNExtractor,
        features_extractor_kwargs=dict(
            num_nodes=MAX_NODES, num_edges=MAX_LINKS,
            hidden_dim=128, num_heads=4, num_topologies=2
        )
    )

    # Check if best model exists
    best_path = "/home/ino/ppo_gnn_improved_best.zip"
    if not os.path.exists(best_path):
        print(f"ERROR: Best model not found at {best_path}")
        print("Available models:")
        for f in os.listdir("/home/ino/"):
            if f.startswith("ppo_gnn") and f.endswith(".zip"):
                print(f"  {f}")
        return

    # ============================================================
    # Experiment 3: Fine-tune GNN Frozen (5k steps)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: Fine-tune GNN Frozen (5k steps)")
    print(f"{'='*70}")

    model_frozen = PPO("MlpPolicy", ab_env, policy_kwargs=policy_kwargs)
    model_frozen = PPO.load("/home/ino/ppo_gnn_improved_best", env=ab_env)
    model_frozen = freeze_gnn_layers(model_frozen)

    t0 = time.time()
    model_frozen.learn(total_timesteps=5000)
    elapsed_frozen = time.time() - t0

    metrics_frozen = evaluate(model_frozen, ab_env, seeds=50)
    gain_frozen = ((metrics_frozen['throughput'] - ab_ospp['throughput']) / ab_ospp['throughput']) * 100

    print(f"\nFine-tune (GNN Frozen) Result ({elapsed_frozen:.0f}s = {elapsed_frozen/60:.1f}min):")
    print(f"  Throughput: {metrics_frozen['throughput']:.1f} Mbps ({gain_frozen:+.1f}% vs OSPF)")
    print(f"  Latency: {metrics_frozen['latency']:.1f}ms")
    print(f"  Loss: {metrics_frozen['loss']:.1f}%")

    model_frozen.save("/home/ino/ppo_gnn_abilene_frozen")
    results = {
        'abilene_ospp': ab_ospp,
        'abilene_optimal': ab_optimal,
        'finetune_frozen': {
            'throughput': metrics_frozen['throughput'],
            'latency': metrics_frozen['latency'],
            'loss': metrics_frozen['loss'],
            'improvement_pct': gain_frozen,
            'training_time_s': elapsed_frozen,
        }
    }

    # ============================================================
    # Experiment 4: Fine-tune All Layers (5k steps)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 4: Fine-tune All Layers (5k steps)")
    print(f"{'='*70}")

    model_full = PPO("MlpPolicy", ab_env, policy_kwargs=policy_kwargs)
    model_full = PPO.load("/home/ino/ppo_gnn_improved_best", env=ab_env)
    model_full = unfreeze_all(model_full)

    t0 = time.time()
    model_full.learn(total_timesteps=5000)
    elapsed_full = time.time() - t0

    metrics_full = evaluate(model_full, ab_env, seeds=50)
    gain_full = ((metrics_full['throughput'] - ab_ospp['throughput']) / ab_ospp['throughput']) * 100

    print(f"\nFine-tune (All Layers) Result ({elapsed_full:.0f}s = {elapsed_full/60:.1f}min):")
    print(f"  Throughput: {metrics_full['throughput']:.1f} Mbps ({gain_full:+.1f}% vs OSPF)")
    print(f"  Latency: {metrics_full['latency']:.1f}ms")
    print(f"  Loss: {metrics_full['loss']:.1f}%")

    model_full.save("/home/ino/ppo_gnn_abilene_full")
    results['finetune_full'] = {
        'throughput': metrics_full['throughput'],
        'latency': metrics_full['latency'],
        'loss': metrics_full['loss'],
        'improvement_pct': gain_full,
        'training_time_s': elapsed_full,
    }

    # ============================================================
    # Experiment 5: From Scratch (5k steps)
    # ============================================================
    print(f"\n{'='*70}")
    print("EXPERIMENT 5: From Scratch (5k steps)")
    print(f"{'='*70}")

    model_scratch = PPO(
        "MlpPolicy", ab_env,
        learning_rate=5e-4, n_steps=64, batch_size=32,
        n_epochs=10, gamma=0.99, ent_coef=0.02, clip_range=0.2,
        verbose=0, seed=42, policy_kwargs=policy_kwargs
    )

    t0 = time.time()
    model_scratch.learn(total_timesteps=5000)
    elapsed_scratch = time.time() - t0

    metrics_scratch = evaluate(model_scratch, ab_env, seeds=50)
    gain_scratch = ((metrics_scratch['throughput'] - ab_ospp['throughput']) / ab_ospp['throughput']) * 100

    print(f"\nFrom Scratch Result ({elapsed_scratch:.0f}s = {elapsed_scratch/60:.1f}min):")
    print(f"  Throughput: {metrics_scratch['throughput']:.1f} Mbps ({gain_scratch:+.1f}% vs OSPF)")
    print(f"  Latency: {metrics_scratch['latency']:.1f}ms")
    print(f"  Loss: {metrics_scratch['loss']:.1f}%")

    model_scratch.save("/home/ino/ppo_gnn_abilene_scratch")
    results['from_scratch'] = {
        'throughput': metrics_scratch['throughput'],
        'latency': metrics_scratch['latency'],
        'loss': metrics_scratch['loss'],
        'improvement_pct': gain_scratch,
        'training_time_s': elapsed_scratch,
    }

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("TRANSFER LEARNING RESULTS — ABILENE")
    print(f"{'='*70}")
    print(f"\n{'Method':<45} {'Tput':>8} {'vs OSPF':>8} {'Latency':>8} {'Loss':>8}")
    print("-" * 80)
    print(f"{'OSPF':<45} {ab_ospp['throughput']:>7.1f}M {'---':>8} {ab_ospp['latency']:>7.1f}ms {ab_ospp['loss']:>7.1f}%")
    print(f"{'Optimal (1/BW)':<45} {ab_optimal['throughput']:>7.1f}M {((ab_optimal['throughput']-ab_ospp['throughput'])/ab_ospp['throughput'])*100:>+7.1f}% {ab_optimal['latency']:>7.1f}ms {ab_optimal['loss']:>7.1f}%")
    print(f"{'Zero-Shot (NSFNET->Abilene)':<45} {'97.3':>7}M {'-7.5':>+7}% {'51.7':>7}ms {'19.5':>7}%")
    print(f"{'Fine-tune GNN Frozen (5k)':<45} {metrics_frozen['throughput']:>7.1f}M {gain_frozen:>+7.1f}% {metrics_frozen['latency']:>7.1f}ms {metrics_frozen['loss']:>7.1f}%")
    print(f"{'Fine-tune All Layers (5k)':<45} {metrics_full['throughput']:>7.1f}M {gain_full:>+7.1f}% {metrics_full['latency']:>7.1f}ms {metrics_full['loss']:>7.1f}%")
    print(f"{'From Scratch (5k)':<45} {metrics_scratch['throughput']:>7.1f}M {gain_scratch:>+7.1f}% {metrics_scratch['latency']:>7.1f}ms {metrics_scratch['loss']:>7.1f}%")

    with open("/home/ino/transfer_exp345_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/ino/transfer_exp345_results.json")

if __name__ == "__main__":
    main()
