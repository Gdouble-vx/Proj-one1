"""
fine_tune_sdn_agent.py — PPO + GNN Fine-Tuning Pipeline สำหรับ Autonomous SDN Routing

กลยุทธ์ (ตาม Handoff Document):
  A) Transfer Learning: โหลด base weights (--base-model) → แช่แข็ง GNN extractor (--freeze)
     → fine-tune policy head ด้วย lr=1e-4 เพียง 2,000–5,000 steps (15–30 นาทีบน ONOS จริง)
  B) 2 โหมด environment:
       --env fast : FastSDNEnv (จำลองเร็ว, ใช้เทรน/pretrain)   [ค่าเริ่มต้น]
       --env onos : CustomSDNEnv (ONOS REST API จริง ผ่าน VM)   [ต้องรัน VM + ONOS ก่อน]
  C) 2 สถาปัตยกรรม:
       --arch gnn : PPO + GNN (SDNGraphFeatureExtractor — GATConv + global mean pool)
       --arch mlp : Vanilla PPO (MLP) — ใช้เป็น baseline เปรียบเทียบ

ตัวอย่างคำสั่ง:
  # 1) Pretrain บน simulator เร็ว ๆ (ประมาณ 10k steps)
  python fine_tune_sdn_agent.py --env fast --arch gnn --total-timesteps 10000 --tag base

  # 2) Fine-tune ต่อกับ ONOS จริง (โหลด base + freeze GNN + lr=1e-4)
  python fine_tune_sdn_agent.py --env onos --arch gnn --vm1-ip 192.168.10.165 \
      --base-model results/ppo_gnn_fast_base --freeze --lr 1e-4 \
      --total-timesteps 3000 --tag onos

  # 3) Vanilla PPO (MLP) สำหรับ benchmark
  python fine_tune_sdn_agent.py --env fast --arch mlp --total-timesteps 10000 --tag vanilla

  # 4) 🔥 ประเมินโมเดล 100k steps เดิมของคุณ (GNN ใน env, action 200 ลิงก์) กับ ONOS จริง
  python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
      --base-model vm_originals/SVs2/ppo_gnn_sdn_model --eval-only --eval-episodes 5

  # 5) Fine-tune ต่อจากโมเดลเดิมของคุณ (obs_mode=gnn, lr=1e-4)
  python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
      --base-model vm_originals/SVs2/ppo_gnn_sdn_model --lr 1e-4 \
      --total-timesteps 3000 --tag resume
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch_geometric.nn import GATConv, global_mean_pool

from custom_sdn_env import CustomSDNEnv
from fast_sdn_env import FastSDNEnv


# ============================================================================
# GNN Feature Extractor (ในฝั่ง policy) — "SDNGraphFeatureExtractor"
# ============================================================================
class SDNGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    รับ observation แบบ raw graph state:
        obs = [node_feat(num_nodes) | edge_attr(max_links*2)]
    แล้วรัน GATConv 2 ชั้น + global_mean_pool → state vector (features_dim,)

    ใช้ได้ทั้ง batch (SB3 ส่ง observations มาเป็น (B, obs_dim))
    """

    def __init__(self, observation_space: gym.Space, num_nodes: int = 14,
                 max_links: int = 50, edge_index=None, features_dim: int = 32,
                 hidden_dim: int = 16):
        super().__init__(observation_space, features_dim)
        self.num_nodes = num_nodes
        self.max_links = max_links
        if edge_index is None:
            edge_index = np.zeros((2, max_links), dtype=np.int64)
        self.register_buffer("edge_index",
                             torch.as_tensor(edge_index, dtype=torch.long))

        # node feature 1 มิติ, edge feature 2 มิติ ([utilization, normalized weight])
        self.conv1 = GATConv(1, hidden_dim, edge_dim=2)
        self.conv2 = GATConv(hidden_dim, features_dim, edge_dim=2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        x = observations[:, : self.num_nodes].reshape(B * self.num_nodes, 1)
        ea = observations[:, self.num_nodes:].reshape(B * self.max_links, 2)

        # ทำ batched graph: edge_index ของแต่ละ sample เลื่อน offset ไปตาม batch
        edge_index = self.edge_index.repeat(1, B)
        offsets = (torch.arange(B, device=observations.device, dtype=torch.long)
                   * self.num_nodes)
        edge_index = edge_index + offsets.repeat_interleave(self.max_links)
        batch = torch.arange(B, device=observations.device).repeat_interleave(self.num_nodes)

        h = F.relu(self.conv1(x, edge_index, ea))
        h = self.conv2(h, edge_index, ea)
        return global_mean_pool(h, batch)          # (B, features_dim)


# ============================================================================
# Callback: ติดตาม FPS / reward / progress และเขียน CSV
# ============================================================================
class ProgressCallback(BaseCallback):
    def __init__(self, log_interval: int = 200, save_dir: str = "results",
                 tag: str = "run", total_steps: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.tag = tag
        self.total_steps = total_steps
        self.start_time = time.time()
        self.ep_returns: list = []
        self.ep_lengths: list = []
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, f"train_{tag}.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestep", "mean_reward", "mean_ep_len", "fps", "time_elapsed"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_returns.append(float(info["episode"]["r"]))
                self.ep_lengths.append(float(info["episode"]["l"]))
        if self.n_calls % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            fps = self.n_calls / elapsed if elapsed > 0 else 0.0
            mean_r = float(np.mean(self.ep_returns[-50:])) if self.ep_returns else float("nan")
            mean_len = float(np.mean(self.ep_lengths[-50:])) if self.ep_lengths else float("nan")
            pct = (100.0 * self.n_calls / self.total_steps) if self.total_steps else 0.0
            print(f"[{pct:5.1f}%] step={self.n_calls:6d} | mean_reward={mean_r:.6f} | "
                  f"mean_ep_len={mean_len:5.1f} | fps={fps:6.0f} | elapsed={elapsed:6.0f}s")
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([self.n_calls, mean_r, mean_len, fps, elapsed])
        return True


# ============================================================================
# Helpers
# ============================================================================
def build_env(args, monitor: bool = True):
    if args.env == "fast":
        env = FastSDNEnv(seed=args.seed, num_nodes=args.num_nodes,
                         num_links=args.num_links)
    else:
        env = CustomSDNEnv(vm1_ip=args.vm1_ip, num_nodes=args.num_nodes,
                           max_links=args.num_links, obs_mode=args.obs_mode)
    return Monitor(env) if monitor else env


def build_policy_kwargs(env, args: argparse.Namespace) -> dict:
    # SB3 ห่อ env ด้วย Monitor → ต้อง unwrap ก่อนอ่าน attributes ของ env จริง
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    # ถ้า GNN อยู่ใน env แล้ว (obs_mode="gnn" = โมเดลเดิมของคุณ) policy ต้องเป็น MLP
    use_gnn = args.arch == "gnn" and getattr(inner, "obs_mode", "raw") != "gnn"
    if args.arch == "gnn" and not use_gnn:
        print("[Warn] env ใช้ obs_mode=gnn (GNN อยู่ใน env แล้ว) → policy เป็น MLP แทน")
    if use_gnn:
        return {
            "features_extractor_class": SDNGraphFeatureExtractor,
            "features_extractor_kwargs": {
                "num_nodes": inner.num_nodes,
                "max_links": inner.max_links,
                "edge_index": inner.edge_index,
                "features_dim": 32,
                "hidden_dim": 16,
            },
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        }
    # Vanilla PPO: default MLP บน observation ตรง ๆ (ไม่ผ่าน GNN)
    return {"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}


def freeze_gnn_extractor(model: PPO, lr: float) -> None:
    """แช่แข็ง GNN extractor → เทรนเฉพาะ policy head (transfer learning)"""
    ex = model.policy.features_extractor
    if not hasattr(ex, "conv1"):
        print("[Freeze] ไม่พบ GNN extractor — ข้ามการ freeze")
        return
    n_params = 0
    for p in ex.parameters():
        p.requires_grad = False
        n_params += 1
    # สร้าง optimizer ใหม่ เฉพาะพารามิเตอร์ที่ยัง train ได้
    model.policy.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.policy.parameters()), lr=lr)
    print(f"[Freeze] แช่แข็ง GNN extractor แล้ว ({n_params} params) — เทรนเฉพาะ policy head")


def build_model(env, args: argparse.Namespace) -> PPO:
    policy_kwargs = build_policy_kwargs(env, args)

    if args.base_model and os.path.exists(args.base_model + ".zip"):
        print(f"[Transfer] โหลด base model จาก {args.base_model}.zip แล้วต่อเทรน...")
        model = PPO.load(args.base_model, env=env, learning_rate=args.lr, device="auto")
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    learning_rate=args.lr, n_steps=args.n_steps,
                    batch_size=args.batch_size, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, seed=args.seed,
                    verbose=0, device="auto")

    if args.freeze and args.arch == "gnn":
        freeze_gnn_extractor(model, args.lr)
    return model


def evaluate(env, model: PPO, episodes: int = 5, base_seed: int = 1000) -> dict:
    """ประเมินโมเดลบน scenario ที่ fix seed → ได้ตัวเลขสำหรับ benchmark/slide"""
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        total_r = 0.0
        info_last = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            total_r += r
            info_last = info
            done = done or trunc
        rows.append({"throughput": info_last.get("throughput", 0),
                     "latency": info_last.get("latency", 0),
                     "packet_loss": info_last.get("packet_loss", 0),
                     "reward": total_r})
        print(f"  ep{ep}: reward={total_r:.6f} | throughput={info_last.get('throughput', 0):.1f} "
              f"Mbps | latency={info_last.get('latency', 0):.2f} ms | "
              f"loss={info_last.get('packet_loss', 0):.4f}")
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="PPO + GNN Fine-Tuning สำหรับ SDN Routing")
    parser.add_argument("--env", choices=["fast", "onos"], default="fast",
                        help="fast=จำลองเร็ว, onos=ONOS จริงผ่าน REST API")
    parser.add_argument("--arch", choices=["gnn", "mlp"], default="gnn",
                        help="gnn=PPO+GNN, mlp=Vanilla PPO")
    parser.add_argument("--obs-mode", choices=["raw", "gnn"], default="raw",
                        help="(เฉพาะ --env onos) raw=policy รัน GNN เอง, gnn=GNN ใน env (โมเดลเดิมของคุณ)")
    parser.add_argument("--eval-only", action="store_true",
                        help="ไม่เทรน — โหลด --base-model แล้วประเมินผลเท่านั้น")
    parser.add_argument("--vm1-ip", default="192.168.10.165", help="IP ของ VM ที่รัน ONOS")
    parser.add_argument("--total-timesteps", type=int, default=5000,
                        help="จำนวน step ที่จะเทรน (fine-tune แนะนำ 2,000–5,000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (fine-tune ใช้ 1e-4)")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--freeze", action="store_true",
                        help="แช่แข็ง GNN extractor (transfer learning)")
    parser.add_argument("--base-model", default=None,
                        help="โหลด weights จากโมเดลก่อนหน้าเพื่อ fine-tune (เช่น results/ppo_gnn_fast_base)")
    parser.add_argument("--tag", default="run1", help="ชื่อ tag สำหรับบันทึกไฟล์")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=14)
    parser.add_argument("--num-links", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"=== PPO + {'GNN' if args.arch == 'gnn' else 'MLP'} Fine-Tuning "
          f"({args.env}) | steps={args.total_timesteps} | lr={args.lr} ===")

    env = build_env(args, monitor=True)
    if args.env == "fast":
        print(f"[Env] {env.env.sim.describe_topology()}")
    else:
        print(f"[Env] ONOS จริงที่ http://{args.vm1_ip}:8181 (obs_mode={args.obs_mode}, "
              f"max_links={args.num_links})")

    if args.eval_only:
        if not args.base_model or not os.path.exists(args.base_model + ".zip"):
            raise SystemExit("--eval-only ต้องระบุ --base-model ที่มีไฟล์ .zip อยู่")
        print(f"[Eval-Only] โหลด {args.base_model}.zip แล้วประเมิน {args.eval_episodes} episodes")
        model = PPO.load(args.base_model, env=env, device="auto")
        eval_env = build_env(args, monitor=False)
        stats = evaluate(eval_env, model, episodes=args.eval_episodes)
        if stats:
            print(f"[Eval-Summary] throughput={stats['throughput']:.1f} Mbps | "
                  f"latency={stats['latency']:.2f} ms | loss={stats['packet_loss']:.4f} | "
                  f"reward={stats['reward']:.6f}")
        return

    model = build_model(env, args)

    # ประมาณการ FPS/เวลาล่วงหน้า (สำหรับสไลด์)
    t0 = time.time()
    obs, _ = env.reset(seed=args.seed)
    model.predict(obs, deterministic=True)
    warm = time.time() - t0
    est_fps = 1.0 / max(warm, 1e-3)
    est_min = args.total_timesteps / max(est_fps, 1e-3) / 60.0
    print(f"[Estimate] ~{est_fps:.0f} steps/s → {args.total_timesteps} steps "
          f"≈ {est_min:.1f} นาที (บน env ปัจจุบัน)")

    callback = ProgressCallback(log_interval=args.log_interval,
                                save_dir=args.save_dir, tag=f"{args.arch}_{args.env}_{args.tag}",
                                total_steps=args.total_timesteps)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    out_path = os.path.join(args.save_dir, f"ppo_{args.arch}_{args.env}_{args.tag}")
    model.save(out_path)
    print(f"[Save] โมเดลบันทึกที่ {out_path}.zip")
    print(f"[Save] training log → {callback.csv_path}")

    # ประเมินผล
    print(f"[Eval] ประเมิน {args.eval_episodes} episodes (seed 1000+):")
    eval_env = build_env(args, monitor=False)
    stats = evaluate(eval_env, model, episodes=args.eval_episodes)
    if stats:
        print(f"[Eval-Summary] throughput={stats['throughput']:.1f} Mbps | "
              f"latency={stats['latency']:.2f} ms | loss={stats['packet_loss']:.4f} | "
              f"reward={stats['reward']:.6f}")


if __name__ == "__main__":
    main()
