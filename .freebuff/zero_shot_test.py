#!/usr/bin/env python3
"""
Zero-shot generalization: Train on NSFNET, test on Abilene
Handles different observation sizes by padding/truncating
"""
import json, time, sys
import numpy as np
sys.path.insert(0, '/home/ino')

from fast_sdn_env import FastSDNEnv
from stable_baselines3 import PPO
from gymnasium import spaces

def pad_observation(obs, target_size):
    """Pad or truncate observation to target size."""
    if len(obs) < target_size:
        padded = np.zeros(target_size, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded
    elif len(obs) > target_size:
        return obs[:target_size]
    return obs

def evaluate(model, env, target_obs_size, seeds=20):
    """Evaluate model with observation padding."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        obs = pad_observation(obs, target_obs_size)
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

def train_ppo(topology, total_timesteps):
    """Train PPO on given topology."""
    env = FastSDNEnv(topology=topology, seed=42)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=256, batch_size=64, n_epochs=10,
        gamma=0.99, verbose=0, seed=42
    )
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    return model, env, elapsed

def main():
    results = {}
    
    print("=" * 60)
    print("ZERO-SHOT GENERALIZATION EXPERIMENT")
    print("=" * 60)
    
    # Get observation sizes
    nsf_env = FastSDNEnv(topology='nsfnet', seed=42)
    ab_env = FastSDNEnv(topology='abilene', seed=42)
    nsf_obs_size = nsf_env.observation_space.shape[0]
    ab_obs_size = ab_env.observation_space.shape[0]
    
    print(f"\nNSFNET obs size: {nsf_obs_size}")
    print(f"Abilene obs size: {ab_obs_size}")
    print(f"NSFNET: {nsf_env.sim.num_nodes} nodes, {nsf_env.sim.num_links} links")
    print(f"Abilene: {ab_env.sim.num_nodes} nodes, {ab_env.sim.num_links} links")
    
    # === Train on NSFNET at different timesteps ===
    for steps in [5000, 10000, 50000]:
        print(f"\n--- Training {steps} steps on NSFNET ---")
        model, env, elapsed = train_ppo('nsfnet', steps)
        print(f"Training time: {elapsed:.1f}s")
        
        # Evaluate on NSFNET
        nsf_metrics = evaluate(model, env, nsf_obs_size, seeds=20)
        print(f"NSFNET: {nsf_metrics['throughput']:.1f} Mbps, {nsf_metrics['latency']:.1f}ms, {nsf_metrics['loss']:.1f}%")
        
        # Zero-shot on Abilene
        ab_metrics = evaluate(model, ab_env, nsf_obs_size, seeds=20)
        print(f"Abilene (zero-shot): {ab_metrics['throughput']:.1f} Mbps, {ab_metrics['latency']:.1f}ms, {ab_metrics['loss']:.1f}%")
        
        results[f'trained_{steps}'] = {
            'nsfnet': nsf_metrics,
            'abilene_zero_shot': ab_metrics,
            'training_time': elapsed
        }
        
        model.save(f"/home/ino/ppo_nsfnet_{steps}")
    
    # === Baselines ===
    print("\n--- Baselines ---")
    
    # NSFNET baselines
    ospp_nsf = evaluate_ospp(nsf_env)
    print(f"NSFNET OSPF: {ospp_nsf['throughput']:.1f} Mbps, {ospp_nsf['latency']:.1f}ms, {ospp_nsf['loss']:.1f}%")
    
    # Abilene baselines
    ospp_ab = evaluate_ospp(ab_env)
    print(f"Abilene OSPF: {ospp_ab['throughput']:.1f} Mbps, {ospp_ab['latency']:.1f}ms, {ospp_ab['loss']:.1f}%")
    
    # Train directly on Abilene for comparison
    print("\n--- Training 10k steps on Abilene (for comparison) ---")
    model_ab, env_ab, elapsed_ab = train_ppo('abilene', 10000)
    ab_direct = evaluate(model_ab, env_ab, ab_obs_size, seeds=20)
    print(f"Abilene (trained): {ab_direct['throughput']:.1f} Mbps, {ab_direct['latency']:.1f}ms, {ab_direct['loss']:.1f}%")
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\n--- NSFNET Results ---")
    print(f"{'Method':<30} {'Tput':>8} {'Latency':>8} {'Loss':>8}")
    print("-" * 56)
    print(f"{'OSPF':<30} {ospp_nsf['throughput']:>7.1f}M {ospp_nsf['latency']:>7.1f}ms {ospp_nsf['loss']:>7.1f}%")
    for steps in [5000, 10000, 50000]:
        m = results[f'trained_{steps}']['nsfnet']
        t = results[f'trained_{steps}']['training_time']
        print(f"{'PPO '+str(steps)+'s ('+str(int(t))+'s)':<30} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}%")
    
    print("\n--- Abilene (Zero-Shot Transfer from NSFNET) ---")
    print(f"{'Method':<30} {'Tput':>8} {'Latency':>8} {'Loss':>8}")
    print("-" * 56)
    print(f"{'OSPF':<30} {ospp_ab['throughput']:>7.1f}M {ospp_ab['latency']:>7.1f}ms {ospp_ab['loss']:>7.1f}%")
    for steps in [5000, 10000, 50000]:
        m = results[f'trained_{steps}']['abilene_zero_shot']
        gain = ((m['throughput'] - ospp_ab['throughput']) / ospp_ab['throughput']) * 100
        print(f"{'Zero-shot '+str(steps)+'s':<30} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}%  ({gain:+.1f}%)")
    print(f"{'Trained 10k (direct)':<30} {ab_direct['throughput']:>7.1f}M {ab_direct['latency']:>7.1f}ms {ab_direct['loss']:>7.1f}%")
    
    # Key insight
    print("\n--- Key Insight ---")
    best_zero = max(results[f'trained_{s}']['abilene_zero_shot']['throughput'] for s in [5000, 10000, 50000])
    print(f"Best zero-shot: {best_zero:.1f} Mbps")
    print(f"OSPF: {ospp_ab['throughput']:.1f} Mbps")
    print(f"Direct training: {ab_direct['throughput']:.1f} Mbps")
    
    results['baselines'] = {'nsfnet_ospp': ospp_nsf, 'abilene_ospp': ospp_ab, 'abilene_direct': ab_direct}
    with open("/home/ino/zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/ino/zero_shot_results.json")

if __name__ == "__main__":
    main()
