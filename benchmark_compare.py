"""
benchmark_compare.py — Benchmark: Dijkstra vs ECMP vs Vanilla PPO (MLP) vs PPO+GNN

ตอบโจทย์อาจารย์: ต้องการตัวเลขเปรียบเทียบเชิงปริมาณว่า PPO+GNN ดีกว่าแบบดั้งเดิมอย่างไร

Metrics:
  - Average Throughput (Mbps)
  - End-to-End Latency (ms)
  - Packet Loss Rate (%)
  - Reward (network power)
  - Convergence Speed (steps ถึง 90% ของ reward สูงสุด — จาก training log)

การใช้งาน:
  # 1) เทรนโมเดลก่อน (ดู fine_tune_sdn_agent.py)
  python fine_tune_sdn_agent.py --env fast --arch gnn --total-timesteps 10000 --tag ppognn
  python fine_tune_sdn_agent.py --env fast --arch mlp --total-timesteps 10000 --tag vanilla

  # 2) รัน benchmark (ทุก method บน scenario เดียวกัน — fix seed)
  python benchmark_compare.py --model-vanilla results/ppo_mlp_fast_vanilla \
                              --model-ppognn results/ppo_gnn_fast_ppognn --episodes 20

  # 2b) ไม่มีโมเดล? ให้ script เทรนสั้น ๆ ให้เอง
  python benchmark_compare.py --quick-train-steps 4000 --episodes 20

  # 3) Zero-Shot Generalization (Topology ใหม่โดยไม่เทรนใหม่)
  python benchmark_compare.py --generalize --model-vanilla ... --model-ppognn ...

หมายเหตุ: baseline (Dijkstra/ECMP) รันได้บน fast env เท่านั้น; ถ้า --env onos จะประเมิน
เฉพาะโมเดล DRL กับ ONOS จริง
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fast_sdn_env import FastSDNEnv

METHODS = ["dijkstra", "ecmp", "vanilla", "ppognn"]
COLORS = {"dijkstra": "#64748b", "ecmp": "#f59e0b", "vanilla": "#0ea5e9", "ppognn": "#8b5cf6"}


# ----------------------------------------------------------------------------
# DRL helpers (import แบบ lazy → baseline รันได้โดยไม่ต้องมี torch/sb3)
# ----------------------------------------------------------------------------
def _import_drl():
    import torch
    from stable_baselines3 import PPO
    from fine_tune_sdn_agent import SDNGraphFeatureExtractor  # noqa: F401 (จำเป็นตอน load)
    return torch, PPO


def _policy_kwargs(arch: str, env):
    from fine_tune_sdn_agent import build_policy_kwargs
    args = SimpleNamespace(arch=arch)
    return build_policy_kwargs(env, args)


def load_or_train_model(arch: str, env, model_path: str, quick_steps: int, seed: int):
    torch, PPO = _import_drl()
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"[Load] {arch}: {model_path}.zip")
        return PPO.load(model_path, device="auto"), False
    if quick_steps > 0:
        print(f"[Train] {arch}: ไม่พบโมเดล → quick-train {quick_steps} steps บน fast env")
        model = PPO("MlpPolicy", env, policy_kwargs=_policy_kwargs(arch, env),
                    learning_rate=3e-4, n_steps=256, batch_size=64,
                    seed=seed, verbose=0, device="auto")
        model.learn(total_timesteps=quick_steps)
        return model, True
    raise FileNotFoundError(
        f"ไม่พบโมเดล {arch} ที่ {model_path}.zip — เทรนก่อนด้วย fine_tune_sdn_agent.py "
        f"หรือใช้ --quick-train-steps")


def set_new_topology(model, env):
    """Zero-shot: สลับ edge_index ของ GNN extractor เป็น topology ใหม่ (ไม่เทรนใหม่)"""
    try:
        torch, _ = _import_drl()
    except ImportError:
        return
    ex = model.policy.features_extractor
    if hasattr(ex, "edge_index"):
        ex.edge_index.data.copy_(torch.as_tensor(env.edge_index, dtype=torch.long))
        print("[Zero-Shot] เปลี่ยน edge_index ของ GNN เป็น topology ใหม่แล้ว (ไม่เทรนใหม่)")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_dijkstra(env, seed):
    env.sample_flows(seed=seed)
    env.reset(seed=seed)
    weights = np.full(env.action_space.shape, 10.0, dtype=np.float32)  # uniform cost = OSPF-like
    info = env.set_weights(weights)
    return {"throughput": info["throughput"], "latency": info["latency"],
            "packet_loss": info["packet_loss"], "reward": info["reward"]}


def run_ecmp(env, seed):
    env.sample_flows(seed=seed)
    env.reset(seed=seed)
    info = env.simulate_ecmp()
    return {"throughput": info["throughput"], "latency": info["latency"],
            "packet_loss": info["packet_loss"], "reward": info["reward"]}


def run_drl(env, model, seed, max_steps=None):
    obs, _ = env.reset(seed=seed)
    done, total_r, info_last = False, 0.0, {}
    max_steps = max_steps or env.max_episode_steps
    steps = 0
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        total_r += float(r)
        info_last = info
        done = done or trunc
        steps += 1
    return {"throughput": info_last.get("throughput", 0),
            "latency": info_last.get("latency", 0),
            "packet_loss": info_last.get("packet_loss", 0),
            "reward": total_r}


def evaluate_method(method: str, env, model, episodes: int, base_seed: int) -> list:
    rows = []
    for ep in range(episodes):
        seed = base_seed + ep
        if method == "dijkstra":
            rows.append(run_dijkstra(env, seed))
        elif method == "ecmp":
            rows.append(run_ecmp(env, seed))
        else:
            rows.append(run_drl(env, model, seed))
    return rows


def aggregate(rows: list) -> dict:
    keys = ["throughput", "latency", "packet_loss", "reward"]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        stats[k] = float(vals.mean())
        stats[k + "_std"] = float(vals.std())
    return stats


# ----------------------------------------------------------------------------
# Convergence (จาก training log CSV)
# ----------------------------------------------------------------------------
def convergence_steps(save_dir: str, name_filter: str) -> str:
    """หาว่าใช้กี่ steps ถึง 90% ของ mean_reward สูงสุด (จาก results/train_*.csv)"""
    files = glob.glob(os.path.join(save_dir, "train_*.csv"))
    files = [f for f in files if name_filter in os.path.basename(f)]
    best = None
    for f in files:
        try:
            with open(f, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [r for r in reader if r.get("mean_reward", "").strip() not in ("", "nan")]
        except Exception:
            continue
        if not rows:
            continue
        steps = [int(r["timestep"]) for r in rows]
        rewards = [float(r["mean_reward"]) for r in rows]
        target = 0.9 * max(rewards)
        for s, r in zip(steps, rewards):
            if r >= target:
                if best is None or s < best:
                    best = s
                break
    return f"{best:,}" if best is not None else "-"


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def plot_results(stats: dict, save_dir: str, title_suffix: str = ""):
    methods = [m for m in METHODS if m in stats]
    labels = {"dijkstra": "Dijkstra / OSPF", "ecmp": "ECMP",
              "vanilla": "Vanilla PPO (MLP)", "ppognn": "PPO + GNN (Proposed)"}
    metrics = [("throughput", "Avg Throughput (Mbps)", "higher is better"),
               ("latency", "Avg End-to-End Latency (ms)", "lower is better"),
               ("packet_loss", "Avg Packet Loss (%)", "lower is better"),
               ("reward", "Avg Reward", "higher is better")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"SDN Routing Benchmark — {title_suffix or 'Fast Simulator'}",
                 fontsize=14, fontweight="bold")
    for ax, (key, ylabel, note) in zip(axes.flat, metrics):
        vals = [stats[m][key] for m in methods]
        errs = [stats[m].get(key + "_std", 0) for m in methods]
        if key == "packet_loss":
            vals = [v * 100 for v in vals]
            errs = [e * 100 for e in errs]
        bars = ax.bar([labels[m] for m in methods], vals, yerr=errs, capsize=6,
                      color=[COLORS[m] for m in methods], alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  ({note})", fontsize=10)
        ax.tick_params(axis="x", rotation=12)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "benchmark_metrics.png")
    fig.savefig(path, dpi=150)
    print(f"[Chart] บันทึก {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Dijkstra vs ECMP vs PPO vs PPO+GNN")
    parser.add_argument("--env", choices=["fast", "onos"], default="fast")
    parser.add_argument("--vm1-ip", default="192.168.10.165")
    parser.add_argument("--methods", default="dijkstra,ecmp,vanilla,ppognn")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="seed สำหรับ quick-train")
    parser.add_argument("--model-vanilla", default=None)
    parser.add_argument("--model-ppognn", default=None)
    parser.add_argument("--quick-train-steps", type=int, default=0,
                        help=">0 = เทรนโมเดลสั้น ๆ ให้เองถ้ายังไม่มี")
    parser.add_argument("--generalize", action="store_true",
                        help="Zero-shot: ทดสอบบน topology ใหม่ (seed ต่าง) โดยไม่เทรนใหม่")
    parser.add_argument("--num-nodes", type=int, default=14)
    parser.add_argument("--num-links", type=int, default=50)
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.env == "onos":
        # baseline ใช้ได้เฉพาะ fast env (ต้องมี simulator)
        methods = [m for m in methods if m not in ("dijkstra", "ecmp")]
        from custom_sdn_env import CustomSDNEnv
        env = CustomSDNEnv(vm1_ip=args.vm1_ip, num_nodes=args.num_nodes,
                           max_links=args.num_links, obs_mode="raw")
        print(f"[Env] ONOS จริง {args.vm1_ip} — ประเมินเฉพาะ DRL: {methods}")
    else:
        env = FastSDNEnv(seed=42, num_nodes=args.num_nodes, num_links=args.num_links)
        print(f"[Env] {env.sim.describe_topology()}")

    models = {}
    if "vanilla" in methods:
        model, _ = load_or_train_model("mlp", env, args.model_vanilla,
                                       args.quick_train_steps, args.seed)
        models["vanilla"] = model
    if "ppognn" in methods:
        model, _ = load_or_train_model("gnn", env, args.model_ppognn,
                                       args.quick_train_steps, args.seed + 1)
        models["ppognn"] = model

    eval_env = env
    if args.generalize:
        if args.env != "fast":
            print("[Warn] --generalize ใช้ได้กับ fast env เท่านั้น — ข้าม")
        else:
            eval_env = FastSDNEnv(seed=999, num_nodes=args.num_nodes,
                                  num_links=args.num_links)
            print(f"[Zero-Shot] Topology ใหม่: {eval_env.sim.describe_topology()}")
            for name, model in models.items():
                set_new_topology(model, eval_env)

    stats = {}
    for method in methods:
        print(f"\n=== Evaluate: {method} ({args.episodes} episodes) ===")
        rows = evaluate_method(method, eval_env, models.get(method),
                               args.episodes, args.base_seed)
        stats[method] = aggregate(rows)
        s = stats[method]
        print(f"  throughput={s['throughput']:.1f} Mbps | latency={s['latency']:.2f} ms | "
              f"loss={s['packet_loss'] * 100:.2f}% | reward={s['reward']:.6f}")

    # convergence จาก training logs
    conv = {}
    if "vanilla" in stats:
        conv["vanilla"] = convergence_steps(args.save_dir, "vanilla")
    if "ppognn" in stats:
        conv["ppognn"] = convergence_steps(args.save_dir, "ppognn")

    # บันทึก CSV/JSON
    csv_path = os.path.join(args.save_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "avg_throughput_mbps", "avg_latency_ms",
                    "avg_packet_loss_pct", "avg_reward", "convergence_steps"])
        for m in stats:
            s = stats[m]
            w.writerow([m, f"{s['throughput']:.2f}", f"{s['latency']:.2f}",
                        f"{s['packet_loss'] * 100:.2f}", f"{s['reward']:.6f}",
                        conv.get(m, "-")])
    print(f"\n[Save] ผลลัพธ์ → {csv_path}")

    json_path = os.path.join(args.save_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({"stats": stats, "convergence": conv}, f, indent=2)

    if args.env == "fast":
        plot_results(stats, args.save_dir,
                     title_suffix="Zero-Shot (Topology ใหม่)" if args.generalize else "Fast Simulator")


if __name__ == "__main__":
    main()
