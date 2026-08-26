#!/usr/bin/env python3
"""Train PPO+GNN 50k steps on asymmetric NSFNET — single run, no loops."""
import json, time, sys
import numpy as np
sys.path.insert(0, '/home/ino')

from train_gnn_v2 import (
    ShapedFastSDNEnv, SDNGraphFeatureExtractor,
    evaluate, evaluate_ospp, evaluate_optimal
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """Print progress every 5000 steps."""
    def __init__(self, eval_env, ospp, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.ospp = ospp
        self.best_tput = 0

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            m = evaluate(self.model, self.eval_env, seeds=20)
            gain = ((m['throughput'] - self.ospp['throughput']) / self.ospp['throughput']) * 100
            tag = ""
            if m['throughput'] > self.best_tput:
                self.best_tput = m['throughput']
                self.model.save("/home/ino/ppo_gnn_50k_best")
                tag = " ★ NEW BEST"
            print(f"  [Step {self.n_calls:>6}] Tput={m['throughput']:.1f}Mbps ({gain:+.1f}%) "
                  f"Lat={m['latency']:.1f}ms Loss={m['loss']:.1f}%{tag}", flush=True)
        return True


def main():
    STEPS = 50000

    print("=" * 60)
    print(f"PPO+GNN Training: {STEPS} steps on Asymmetric NSFNET")
    print("=" * 60)

    env = ShapedFastSDNEnv(topology='nsfnet', seed=42)
    eval_env = ShapedFastSDNEnv(topology='nsfnet', seed=99)

    # Baseline
    ospp = evaluate_ospp(env, seeds=50)
    print(f"OSPF baseline: {ospp['throughput']:.1f} Mbps, {ospp['latency']:.1f}ms, {ospp['loss']:.1f}%")

    # Policy
    policy_kwargs = dict(
        features_extractor_class=SDNGraphFeatureExtractor,
        features_extractor_kwargs=dict(
            num_nodes=env.num_nodes,
            num_edges=env.num_links,
            hidden_dim=128,
            num_heads=4
        )
    )

    model = PPO(
        "MlpPolicy", env,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.02,
        clip_range=0.2,
        verbose=0,
        seed=42,
        policy_kwargs=policy_kwargs
    )

    callback = ProgressCallback(eval_env=eval_env, ospp=ospp)

    print(f"\nStarting training ({STEPS} steps)...")
    t0 = time.time()
    model.learn(total_timesteps=STEPS, callback=callback)
    elapsed = time.time() - t0

    # Final eval (100 seeds)
    print(f"\nTraining done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("Final evaluation (100 seeds)...")
    metrics = evaluate(model, eval_env, seeds=100)
    gain = ((metrics['throughput'] - ospp['throughput']) / ospp['throughput']) * 100

    model.save("/home/ino/ppo_gnn_nsfnet_50k")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"  Throughput: {metrics['throughput']:.2f} Mbps ({gain:+.2f}% vs OSPF)")
    print(f"  Latency:    {metrics['latency']:.2f}ms")
    print(f"  Loss:       {metrics['loss']:.2f}%")
    print(f"  OSPF:       {ospp['throughput']:.2f} Mbps")
    print(f"  Training:   {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved:      ppo_gnn_nsfnet_50k.zip + ppo_gnn_50k_best.zip")

    results = {
        'ospp': ospp,
        'gnn_50k': metrics,
        'gnn_50k_improvement_pct': gain,
        'training_time_s': elapsed,
    }
    with open("/home/ino/gnn_50k_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results:    gnn_50k_results.json")


if __name__ == "__main__":
    main()
