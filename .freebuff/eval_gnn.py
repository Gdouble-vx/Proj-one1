#!/usr/bin/env python3
"""Evaluate GNN models on NSFNET and Abilene"""
import json, time, sys
import numpy as np
import torch
sys.path.insert(0, '/home/ino')

from fast_sdn_env import FastSDNEnv
from stable_baselines3 import PPO

def evaluate(model, env, seeds=20):
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
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
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        env.reset(seed=s)
        m = env.set_weights(np.ones(env.max_links, dtype=np.float32))
        throughputs.append(m['throughput'])
        latencies.append(m['latency'])
        losses.append(m['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }

def main():
    print("=" * 60)
    print("GNN Model Evaluation")
    print("=" * 60)
    
    # === NSFNET ===
    print("\n--- NSFNET ---")
    env_nsf = FastSDNEnv(topology='nsfnet', seed=42)
    ospp_nsf = evaluate_ospp(env_nsf)
    print(f"OSPF: {ospp_nsf['throughput']:.1f} Mbps, {ospp_nsf['latency']:.1f}ms, {ospp_nsf['loss']:.1f}%")
    
    results = {'nsfnet_ospp': ospp_nsf}
    
    for steps in [5000, 10000]:
        try:
            model = PPO.load(f"/home/ino/ppo_gnn_nsfnet_{steps}", env=env_nsf)
            metrics = evaluate(model, env_nsf)
            gain = ((metrics['throughput'] - ospp_nsf['throughput']) / ospp_nsf['throughput']) * 100
            print(f"GNN {steps}k: {metrics['throughput']:.1f} Mbps ({gain:+.1f}%), {metrics['latency']:.1f}ms, {metrics['loss']:.1f}%")
            results[f'gnn_{steps}'] = metrics
        except Exception as e:
            print(f"GNN {steps}k failed: {e}")
    
    # === Abilene Zero-Shot ===
    print("\n--- Abilene (Zero-Shot Transfer) ---")
    env_ab = FastSDNEnv(topology='abilene', seed=42)
    ospp_ab = evaluate_ospp(env_ab)
    print(f"OSPF: {ospp_ab['throughput']:.1f} Mbps, {ospp_ab['latency']:.1f}ms, {ospp_ab['loss']:.1f}%")
    results['abilene_ospp'] = ospp_ab
    
    for steps in [5000, 10000]:
        try:
            model = PPO.load(f"/home/ino/ppo_gnn_nsfnet_{steps}", env=env_ab)
            metrics = evaluate(model, env_ab)
            gain = ((metrics['throughput'] - ospp_ab['throughput']) / ospp_ab['throughput']) * 100
            print(f"Zero-shot {steps}k: {metrics['throughput']:.1f} Mbps ({gain:+.1f}%), {metrics['latency']:.1f}ms, {metrics['loss']:.1f}%")
            results[f'zero_shot_{steps}'] = metrics
        except Exception as e:
            print(f"Zero-shot {steps}k failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<35} {'Tput':>8} {'Latency':>8} {'Loss':>8}")
    print("-" * 60)
    print(f"{'NSFNET OSPF':<35} {ospp_nsf['throughput']:>7.1f}M {ospp_nsf['latency']:>7.1f}ms {ospp_nsf['loss']:>7.1f}%")
    for s in [5000, 10000]:
        if f'gnn_{s}' in results:
            m = results[f'gnn_{s}']
            print(f"{'PPO+GNN '+str(s)+'k':<35} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}%")
    print(f"\n{'Abilene OSPF':<35} {ospp_ab['throughput']:>7.1f}M {ospp_ab['latency']:>7.1f}ms {ospp_ab['loss']:>7.1f}%")
    for s in [5000, 10000]:
        key = f'zero_shot_{s}'
        if key in results:
            m = results[key]
            gain = ((m['throughput'] - ospp_ab['throughput']) / ospp_ab['throughput']) * 100
            print(f"{'Zero-shot '+str(s)+'k':<35} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}% ({gain:+.1f}%)")
    
    with open("/home/ino/gnn_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/ino/gnn_eval_results.json")

if __name__ == "__main__":
    main()
