#!/usr/bin/env python3
"""
Multi-step evaluation: run 200 steps per episode with different flow patterns
This better simulates real network behavior over time.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/home/ino')

from train_gnn_v2 import ShapedFastSDNEnv, SDNGraphFeatureExtractor
from stable_baselines3 import PPO

def eval_multistep(model, env, num_episodes=20, steps_per_episode=200):
    """Run model for multiple steps per episode, accumulate metrics."""
    all_throughputs, all_latencies, all_losses = [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        ep_throughputs, ep_latencies, ep_losses = [], [], []

        for step in range(steps_per_episode):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.full(env.max_links, 10.0, dtype=np.float32)  # OSPF uniform

            obs, reward, done, trunc, info = env.step(action)
            ep_throughputs.append(info['throughput'])
            ep_latencies.append(info['latency'])
            ep_losses.append(info['packet_loss'])

            if done:
                obs, _ = env.reset(seed=ep * 1000 + step + 1)

        all_throughputs.append(np.mean(ep_throughputs))
        all_latencies.append(np.mean(ep_latencies))
        all_losses.append(np.mean(ep_losses))

    return {
        'throughput': np.mean(all_throughputs),
        'throughput_std': np.std(all_throughputs),
        'latency': np.mean(all_latencies),
        'latency_std': np.std(all_latencies),
        'loss': np.mean(all_losses) * 100,
        'loss_std': np.std(all_losses) * 100,
    }


def eval_optimal_multistep(env, num_episodes=20, steps_per_episode=200):
    """Evaluate optimal (1/BW) baseline over multi-step episodes."""
    all_throughputs, all_latencies, all_losses = [], [], []
    caps = np.array(env.sim.link_capacities, dtype=np.float32)
    optimal_weights = np.clip(1000.0 / caps, 1.0, 100.0)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        ep_throughputs, ep_latencies, ep_losses = [], [], []

        for step in range(steps_per_episode):
            obs, reward, done, trunc, info = env.step(optimal_weights)
            ep_throughputs.append(info['throughput'])
            ep_latencies.append(info['latency'])
            ep_losses.append(info['packet_loss'])

            if done:
                obs, _ = env.reset(seed=ep * 1000 + step + 1)

        all_throughputs.append(np.mean(ep_throughputs))
        all_latencies.append(np.mean(ep_latencies))
        all_losses.append(np.mean(ep_losses))

    return {
        'throughput': np.mean(all_throughputs),
        'throughput_std': np.std(all_throughputs),
        'latency': np.mean(all_latencies),
        'latency_std': np.std(all_latencies),
        'loss': np.mean(all_losses) * 100,
        'loss_std': np.std(all_losses) * 100,
    }


def main():
    print("=" * 70)
    print("MULTI-STEP EVALUATION (200 steps × 20 episodes = 4000 steps)")
    print("=" * 70)

    env = ShapedFastSDNEnv(topology='nsfnet', seed=42)
    policy_kwargs = dict(
        features_extractor_class=SDNGraphFeatureExtractor,
        features_extractor_kwargs=dict(
            num_nodes=env.num_nodes, num_edges=env.num_links,
            hidden_dim=128, num_heads=4
        )
    )

    # OSPF
    print("\n[1/4] OSPF baseline...")
    t0 = time.time()
    ospp = eval_multistep(None, env, num_episodes=20, steps_per_episode=200)
    print(f"  Throughput: {ospp['throughput']:.2f} ± {ospp['throughput_std']:.2f} Mbps")
    print(f"  Latency:    {ospp['latency']:.2f} ± {ospp['latency_std']:.2f} ms")
    print(f"  Loss:       {ospp['loss']:.2f} ± {ospp['loss_std']:.2f}%")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Optimal
    print("\n[2/4] Optimal 1/BW baseline...")
    t0 = time.time()
    optimal = eval_optimal_multistep(env, num_episodes=20, steps_per_episode=200)
    print(f"  Throughput: {optimal['throughput']:.2f} ± {optimal['throughput_std']:.2f} Mbps")
    print(f"  Latency:    {optimal['latency']:.2f} ± {optimal['latency_std']:.2f} ms")
    print(f"  Loss:       {optimal['loss']:.2f} ± {optimal['loss_std']:.2f}%")
    print(f"  Time: {time.time()-t0:.0f}s")

    # PPO+GNN Best
    print("\n[3/4] PPO+GNN Best (5k checkpoint)...")
    model_best = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
    model_best = PPO.load("/home/ino/ppo_gnn_best", env=env)
    t0 = time.time()
    gnn_best = eval_multistep(model_best, env, num_episodes=20, steps_per_episode=200)
    print(f"  Throughput: {gnn_best['throughput']:.2f} ± {gnn_best['throughput_std']:.2f} Mbps")
    print(f"  Latency:    {gnn_best['latency']:.2f} ± {gnn_best['latency_std']:.2f} ms")
    print(f"  Loss:       {gnn_best['loss']:.2f} ± {gnn_best['loss_std']:.2f}%")
    print(f"  Time: {time.time()-t0:.0f}s")

    # PPO+GNN 10k
    print("\n[4/4] PPO+GNN 10k...")
    model_10k = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
    model_10k = PPO.load("/home/ino/ppo_gnn_nsfnet_10000", env=env)
    t0 = time.time()
    gnn_10k = eval_multistep(model_10k, env, num_episodes=20, steps_per_episode=200)
    print(f"  Throughput: {gnn_10k['throughput']:.2f} ± {gnn_10k['throughput_std']:.2f} Mbps")
    print(f"  Latency:    {gnn_10k['latency']:.2f} ± {gnn_10k['latency_std']:.2f} ms")
    print(f"  Loss:       {gnn_10k['loss']:.2f} ± {gnn_10k['loss_std']:.2f}%")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Summary
    def pct(a, b):
        return ((a - b) / b) * 100

    print("\n" + "=" * 70)
    print("MULTI-STEP RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Tput (Mbps)':>14} {'vs OSPF':>10} {'Latency (ms)':>14} {'Loss (%)':>10}")
    print("-" * 83)
    print(f"{'OSPF (hop-count)':<35} {ospp['throughput']:>13.2f} {'---':>10} {ospp['latency']:>13.2f} {ospp['loss']:>9.2f}")
    print(f"{'Optimal (1/BW)':<35} {optimal['throughput']:>13.2f} {pct(optimal['throughput'], ospp['throughput']):>+9.2f}% {optimal['latency']:>13.2f} {optimal['loss']:>9.2f}")
    print(f"{'PPO+GNN Best (5k ckpt)':<35} {gnn_best['throughput']:>13.2f} {pct(gnn_best['throughput'], ospp['throughput']):>+9.2f}% {gnn_best['latency']:>13.2f} {gnn_best['loss']:>9.2f}")
    print(f"{'PPO+GNN 10k':<35} {gnn_10k['throughput']:>13.2f} {pct(gnn_10k['throughput'], ospp['throughput']):>+9.2f}% {gnn_10k['latency']:>13.2f} {gnn_10k['loss']:>9.2f}")

    # Improvement over OSPF
    print(f"\n--- Improvement over OSPF ---")
    print(f"PPO+GNN Best: Throughput {pct(gnn_best['throughput'], ospp['throughput']):+.2f}%, "
          f"Latency {pct(gnn_best['latency'], ospp['latency']):+.2f}%, "
          f"Loss {pct(gnn_best['loss'], ospp['loss']):+.2f}%")

    results = {
        'ospp': ospp, 'optimal': optimal,
        'gnn_best': gnn_best, 'gnn_10k': gnn_10k,
        'gnn_best_vs_ospp': {
            'throughput_pct': pct(gnn_best['throughput'], ospp['throughput']),
            'latency_pct': pct(gnn_best['latency'], ospp['latency']),
            'loss_pct': pct(gnn_best['loss'], ospp['loss']),
        }
    }
    with open("/home/ino/gnn_v2_multistep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to /home/ino/gnn_v2_multistep.json")


if __name__ == "__main__":
    main()
