#!/usr/bin/env python3
"""
Train PPO with reward shaping:
- Bonus for high throughput
- Penalty for using narrow links
- Penalty for packet loss
"""
import json, time, sys, os
import numpy as np
sys.path.insert(0, '/home/ino')

from fast_sdn_env import FastSDNEnv
from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym

class RewardShapedEnv(gym.Wrapper):
    """Wrapper that adds reward shaping to encourage wide-link routing."""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_capacities = env.sim.link_capacities
        self.narrow_mask = self.link_capacities < 30  # Narrow links
        self.wide_mask = self.link_capacities >= 100  # Wide links
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current weights and utilization
        weights = self.env.weights
        utilization = self.env.last_metrics.get('link_utilization', np.zeros(len(weights)))
        
        # Reward shaping:
        # 1. Base reward (throughput/latency)
        shaped_reward = reward
        
        # 2. Penalty for high weights on wide links (discourage using wide links with high cost)
        wide_util = utilization[self.wide_mask].mean() if self.wide_mask.any() else 0
        
        # 3. Bonus for routing through wide links
        if wide_util > 0.1:  # If wide links are being used
            shaped_reward += 0.001 * wide_util  # Small bonus
        
        # 4. Penalty for packet loss
        if self.env.last_metrics.get('packet_loss', 0) > 0.1:
            shaped_reward -= 0.01
        
        return obs, shaped_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def evaluate(model, env, obs_size, seeds=20):
    """Evaluate model with fixed observation padding."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        obs = pad_obs(obs, obs_size)
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

def pad_obs(obs, size):
    if len(obs) < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[:len(obs)] = obs
        return padded
    return obs[:size]

def main():
    results = {}
    
    print("=" * 60)
    print("REWARD-SHAPED PPO TRAINING")
    print("=" * 60)
    
    env = FastSDNEnv(topology='nsfnet', seed=42)
    shaped_env = RewardShapedEnv(env)
    obs_size = env.observation_space.shape[0]
    
    print(f"NSFNET: {env.sim.num_nodes} nodes, {env.sim.num_links} links")
    print(f"Observation size: {obs_size}")
    print(f"Narrow links (<30 Mbps): {(env.sim.link_capacities < 30).sum()}")
    print(f"Wide links (>=100 Mbps): {(env.sim.link_capacities >= 100).sum()}")
    
    # Train at different timesteps
    for steps in [5000, 10000, 50000]:
        print(f"\n--- Training {steps} steps with reward shaping ---")
        model = PPO(
            "MlpPolicy", shaped_env,
            learning_rate=3e-4,
            n_steps=256, batch_size=64, n_epochs=10,
            gamma=0.99, verbose=0, seed=42
        )
        
        t0 = time.time()
        model.learn(total_timesteps=steps)
        elapsed = time.time() - t0
        
        # Evaluate on plain env
        metrics = evaluate(model, env, obs_size, seeds=20)
        
        gain = ((metrics['throughput'] - 99.9) / 99.9) * 100  # vs OSPF baseline
        print(f"Training: {elapsed:.1f}s")
        print(f"Result: {metrics['throughput']:.1f} Mbps ({gain:+.1f}% vs OSPF), "
              f"{metrics['latency']:.1f}ms, {metrics['loss']:.1f}%")
        
        results[f'shaped_{steps}'] = metrics
        results[f'shaped_{steps}']['training_time'] = elapsed
        results[f'shaped_{steps}']['improvement'] = gain
        
        model.save(f"/home/ino/ppo_shaped_{steps}")
    
    # Also test on Abilene (zero-shot)
    print("\n--- Zero-Shot Transfer to Abilene ---")
    ab_env = FastSDNEnv(topology='abilene', seed=42)
    
    for steps in [5000, 10000, 50000]:
        try:
            model = PPO.load(f"/home/ino/ppo_shaped_{steps}", env=shaped_env)
            metrics = evaluate(model, ab_env, obs_size, seeds=20)
            gain = ((metrics['throughput'] - 105.2) / 105.2) * 100
            print(f"Zero-shot {steps}: {metrics['throughput']:.1f} Mbps ({gain:+.1f}% vs Abilene OSPF)")
            results[f'shaped_{steps}_abilene'] = metrics
        except Exception as e:
            print(f"Zero-shot {steps} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Method':<35} {'Tput':>8} {'Latency':>8} {'Loss':>8} {'vs OSPF':>8}")
    print("-" * 68)
    print(f"{'OSPF (baseline)':<35} {'99.9':>8} {'30.5ms':>8} {'17.9%':>8} {'---':>8}")
    
    for steps in [5000, 10000, 50000]:
        m = results[f'shaped_{steps}']
        t = m['training_time']
        print(f"{'PPO shaped '+str(steps)+'s ('+str(int(t))+'s)':<35} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}% {m['improvement']:>+7.1f}%")
    
    print(f"\n--- Zero-Shot Transfer to Abilene ---")
    print(f"{'Abilene OSPF':<35} {'105.2':>8} {'26.3ms':>8} {'13.8%':>8} {'---':>8}")
    for steps in [5000, 10000, 50000]:
        key = f'shaped_{steps}_abilene'
        if key in results:
            m = results[key]
            gain = ((m['throughput'] - 105.2) / 105.2) * 100
            print(f"{'Zero-shot '+str(steps)+'s':<35} {m['throughput']:>7.1f}M {m['latency']:>7.1f}ms {m['loss']:>7.1f}% {gain:>+7.1f}%")
    
    with open("/home/ino/reward_shaped_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /home/ino/reward_shaped_results.json")

if __name__ == "__main__":
    main()
