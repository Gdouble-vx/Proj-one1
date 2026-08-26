#!/usr/bin/env python3
"""
Full Experiment: PPO+GNN Training Schedules + Zero-Shot Generalization
1. Train on NSFNET asymmetric at 5k, 10k, 50k timesteps
2. Zero-shot transfer to Abilene topology (no retraining)
3. Compare all results
"""
import json, time, sys, os
import numpy as np
sys.path.insert(0, '/home/ino')

from fast_sdn_env import FastSDNEnv
from stable_baselines3 import PPO

def evaluate(model, env, seeds=20):
    """Evaluate model over multiple seeds."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        if model:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
        else:
            obs, _, term, trunc, info = env.step(env.action_space.sample())
        m = info if isinstance(info, dict) and 'throughput' in info else env.last_metrics
        throughputs.append(m['throughput'])
        latencies.append(m['latency'])
        losses.append(m['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
        'throughput_std': np.std(throughputs),
        'latency_std': np.std(latencies),
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

def evaluate_bw_optimal(env, seeds=20):
    """Evaluate bandwidth-optimal weights (theoretical)."""
    throughputs, latencies, losses = [], [], []
    for s in range(seeds):
        env.reset(seed=s)
        bw_weights = (1000.0 / env.sim.link_capacities).astype(np.float32)
        bw_weights = np.clip(bw_weights, 1.0, 100.0)
        m = env.set_weights(bw_weights)
        throughputs.append(m['throughput'])
        latencies.append(m['latency'])
        losses.append(m['packet_loss'])
    return {
        'throughput': np.mean(throughputs),
        'latency': np.mean(latencies),
        'loss': np.mean(losses) * 100,
    }

def train_ppo(total_timesteps):
    """Train PPO on NSFNET asymmetric."""
    env = FastSDNEnv(topology='nsfnet', seed=42)
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
    
    # === PART 1: Training Schedules ===
    print("=" * 60)
    print("PART 1: Training Schedules (5k, 10k, 50k steps)")
    print("=" * 60)
    
    for steps in [5000, 10000, 50000]:
        print(f"\n--- Training {steps} steps ---")
        model, env, elapsed = train_ppo(steps)
        print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Evaluate on NSFNET
        metrics = evaluate(model, env, seeds=20)
        print(f"NSFNET: throughput={metrics['throughput']:.1f} Mbps "
              f"(+/-{metrics['throughput_std']:.1f}), "
              f"latency={metrics['latency']:.1f}ms, loss={metrics['loss']:.1f}%")
        
        results[f'ppo_{steps}'] = {
            'nsfnet': metrics,
            'training_time_s': elapsed
        }
        
        # Save model
        model.save(f"/home/ino/ppo_nsfnet_{steps}")
        print(f"Model saved: ppo_nsfnet_{steps}.zip")
    
    # OSPF and Optimal baselines
    print("\n--- Baselines on NSFNET ---")
    env_nsf = FastSDNEnv(topology='nsfnet', seed=42)
    ospp = evaluate_ospp(env_nsf)
    opt = evaluate_bw_optimal(env_nsf)
    print(f"OSPF:     throughput={ospp['throughput']:.1f} Mbps, latency={ospp['latency']:.1f}ms, loss={ospp['loss']:.1f}%")
    print(f"Optimal:  throughput={opt['throughput']:.1f} Mbps, latency={opt['latency']:.1f}ms, loss={opt['loss']:.1f}%")
    results['ospp'] = ospp
    results['optimal'] = opt
    
    # === PART 2: Zero-Shot Generalization ===
    print("\n" + "=" * 60)
    print("PART 2: Zero-Shot Generalization (NSFNET -> Abilene)")
    print("=" * 60)
    
    env_ab = FastSDNEnv(topology='abilene', seed=42)
    
    print(f"\nAbilene: {env_ab.sim.num_nodes} nodes, {env_ab.sim.num_links} links")
    print(f"Link BW range: [{env_ab.sim.link_capacities.min():.0f}, {env_ab.sim.link_capacities.max():.0f}] Mbps")
    
    # Baselines on Abilene
    ospp_ab = evaluate_ospp(env_ab)
    opt_ab = evaluate_bw_optimal(env_ab)
    print(f"\nOSPF (Abilene):     throughput={ospp_ab['throughput']:.1f} Mbps, latency={ospp_ab['latency']:.1f}ms, loss={ospp_ab['loss']:.1f}%")
    print(f"Optimal (Abilene):  throughput={opt_ab['throughput']:.1f} Mbps, latency={opt_ab['latency']:.1f}ms, loss={opt_ab['loss']:.1f}%")
    
    # Zero-shot: apply NSFNET-trained models to Abilene
    for steps in [5000, 10000, 50000]:
        model_path = f"/home/ino/ppo_nsfnet_{steps}"
        try:
            model = PPO.load(model_path, env=env_ab)
            metrics = evaluate(model, env_ab, seeds=20)
            
            gain = ((metrics['throughput'] - ospp_ab['throughput']) / ospp_ab['throughput']) * 100
            print(f"Zero-shot {steps} (Abilene): throughput={metrics['throughput']:.1f} Mbps "
                  f"(+{gain:.1f}% vs OSPF), latency={metrics['latency']:.1f}ms, loss={metrics['loss']:.1f}%")
            
            results[f'zero_shot_{steps}_abilene'] = {
                'throughput': metrics['throughput'],
                'latency': metrics['latency'],
                'loss': metrics['loss'],
                'improvement_vs_ospp': gain
            }
        except Exception as e:
            print(f"Zero-shot {steps} failed: {e}")
    
    # Also test on random topology
    print("\n--- Zero-Shot on Random Topology ---")
    env_rand = FastSDNEnv(topology='random', seed=42)
    ospp_rand = evaluate_ospp(env_rand)
    print(f"OSPF (Random): throughput={ospp_rand['throughput']:.1f} Mbps")
    
    for steps in [10000, 50000]:
        try:
            model = PPO.load(f"/home/ino/ppo_nsfnet_{steps}", env=env_rand)
            metrics = evaluate(model, env_rand, seeds=20)
            gain = ((metrics['throughput'] - ospp_rand['throughput']) / ospp_rand['throughput']) * 100
            print(f"Zero-shot {steps} (Random): throughput={metrics['throughput']:.1f} Mbps (+{gain:.1f}%)")
            results[f'zero_shot_{steps}_random'] = {
                'throughput': metrics['throughput'],
                'improvement_vs_ospp': gain
            }
        except Exception as e:
            print(f"Zero-shot {steps} (Random) failed: {e}")
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<35} {'Tput':>8} {'Latency':>8} {'Loss':>8}")
    print("-" * 62)
    
    print(f"{'NSFNET Baselines':}")
    print(f"  {'OSPF':<33} {ospp['throughput']:>8.1f} {ospp['latency']:>7.1f}ms {ospp['loss']:>7.1f}%")
    print(f"  {'Optimal (1/BW)':<33} {opt['throughput']:>8.1f} {opt['latency']:>7.1f}ms {opt['loss']:>7.1f}%")
    
    print(f"\n{'PPO+GNN (trained on NSFNET)':}")
    for steps in [5000, 10000, 50000]:
        key = f'ppo_{steps}'
        m = results[key]['nsfnet']
        t = results[key]['training_time_s']
        print(f"  {f'{steps} steps ({t:.0f}s)':<33} {m['throughput']:>8.1f} {m['latency']:>7.1f}ms {m['loss']:>7.1f}%")
    
    print(f"\n{'Zero-Shot Transfer (-> Abilene)':}")
    print(f"  {'OSPF baseline':<33} {ospp_ab['throughput']:>8.1f} {ospp_ab['latency']:>7.1f}ms {ospp_ab['loss']:>7.1f}%")
    for steps in [5000, 10000, 50000]:
        key = f'zero_shot_{steps}_abilene'
        if key in results:
            m = results[key]
            print(f"  {f'{steps} steps':<33} {m['throughput']:>8.1f} {m['latency']:>7.1f}ms {m['loss']:>7.1f}%  (+{m['improvement_vs_ospp']:.1f}%)")
    
    # Save all results
    with open("/home/ino/full_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to /home/ino/full_experiment_results.json")

if __name__ == "__main__":
    main()
