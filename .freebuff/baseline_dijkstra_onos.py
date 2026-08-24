"""Dijkstra/OSPF baseline on real ONOS — same protocol as PPO+GNN eval.

OSPF = equal-cost link weights (default ONOS, no annotations) + Dijkstra shortest path.
Here we drive CustomSDNEnv with a fixed action = ones(200) (weight 1.0 = default on all
links) so the network stays at OSPF state, and measure real metrics through the same
loop as evaluate() in fine_tune_sdn_agent.py: 5 episodes x 200 steps, seeds 1000-1004.
"""
import json
import numpy as np
from custom_sdn_env import CustomSDNEnv

EPISODES = 5
STEPS_PER_EP = 200
BASE_SEED = 1000

env = CustomSDNEnv(vm1_ip="192.168.10.165", num_nodes=14, max_links=21,
                   obs_mode="gnn", use_real_metrics=True, step_delay=2.5)

rows = []
for ep in range(EPISODES):
    obs, _ = env.reset(seed=BASE_SEED + ep)
    total_r = 0.0
    info_last = {}
    for s in range(STEPS_PER_EP):
        action = np.ones(env.action_space.shape, dtype=np.float32)  # OSPF equal weights
        obs, r, done, trunc, info = env.step(action)
        total_r += float(r)
        info_last = info
    rows.append({"throughput": info_last.get("throughput", 0),
                 "latency": info_last.get("latency", 0),
                 "packet_loss": info_last.get("packet_loss", 0),
                 "reward": total_r})
    print(f"  ep{ep}: reward={total_r:.6f} | throughput={info_last.get('throughput', 0):.1f} "
          f"Mbps | latency={info_last.get('latency', 0):.2f} ms | "
          f"loss={info_last.get('packet_loss', 0):.4f}", flush=True)

stats = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
print("[Dijkstra-Summary] " + json.dumps(stats), flush=True)
with open("results_dijkstra_onos.json", "w") as f:
    json.dump({"rows": rows, "summary": stats}, f, indent=2)
print("saved results_dijkstra_onos.json", flush=True)
env.close()
