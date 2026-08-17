# Autonomous SDN Routing Optimization — PPO + GNN (ONOS)

ระบบควบคุมเส้นทางอัตโนมัติบน SDN: PPO + Graph Neural Network ปรับค่าน้ำหนักลิงก์
แบบ Closed-Loop ผ่าน REST API ของ ONOS Controller (Mininet 14 โหนด / 50 ลิงก์)

## ไฟล์ในโปรเจกต์

| ไฟล์ | หน้าที่ |
|---|---|
| `network_sim.py` | เครื่องจำลองเครือข่ายเร็วมาก (NumPy ล้วน): topology, Dijkstra, ECMP, metrics |
| `fast_sdn_env.py` | `FastSDNEnv` — gymnasium env รอบ simulator ใช้เทรน/pretrain เร็ว |
| `custom_sdn_env.py` | `CustomSDNEnv` — env ต่อ ONOS จริงผ่าน REST API (ย้ายมาจาก `import gymnasium as gym.py`) |
| `fine_tune_sdn_agent.py` | Pipeline หลัก: PPO+GNN, freeze layer, transfer learning, `--env fast\|onos` |
| `benchmark_compare.py` | Benchmark: Dijkstra vs ECMP vs Vanilla PPO vs PPO+GNN + กราฟ |
| `presentation_results.md` | ข้อมูลสรุปสำหรับสไลด์ Canva (ตาราง + ข้อความ) |
| `vm_originals/` | โค้ด + โมเดลจริงที่ extract ออกจาก VM (SVs1 = Mininet/ONOS, SVs2 = AI brain) |
| `results/` | output: โมเดล `.zip`, training log `.csv`, benchmark results + charts |

> หมายเหตุ: `custom_sdn_env.py` = merge ระหว่าง env เดิมของคุณ (`import gymnasium as gym.py`)
> กับเวอร์ชันจริงใน VM (`vm_originals/SVs2/sdn_network_env.py`) — รวมการส่ง weight แบบ delta (>2.0),
> timeout 5s, sleep 2.5s/step ตามของจริง + เพิ่ม `obs_mode="raw"` สำหรับ policy-side GNN

## ติดตั้ง (บนเครื่อง ML / VM ที่ใช้เทรน)

```bash
pip install numpy matplotlib gymnasium torch torch-geometric stable-baselines3 requests
```

> torch_geometric จำเป็นสำหรับ `SDNGraphFeatureExtractor` (GATConv) และ `CustomSDNEnv`

## ขั้นตอนการทำงาน (Workflow)

```
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│ 1. Pretrain (fast)  │ ───► │ 2. Fine-tune (ONOS)  │ ───► │ 3. Benchmark     │
│    simulator เร็ว    │      │    transfer + freeze  │      │    + graphs      │
│    10k steps         │      │    2k–5k steps        │      │    + slides      │
└─────────────────────┘      └──────────────────────┘      └──────────────────┘
```

### 1) Pretrain บน FastSDNEnv (เร็วมาก)

```bash
python fine_tune_sdn_agent.py --env fast --arch gnn --total-timesteps 10000 --tag base
# Vanilla PPO (MLP) สำหรับเปรียบเทียบ
python fine_tune_sdn_agent.py --env fast --arch mlp --total-timesteps 10000 --tag vanilla
```

### 2) Fine-tune กับ ONOS จริง

ก่อนรัน: เปิด VM + ONOS + Mininet แล้วตรวจว่า REST API เข้าถึงได้

```bash
curl -u onos:rocks http://<vm1-ip>:8181/onos/v1/devices   # ควรได้ JSON
```

**แบบ A — โมเดลเดิมของคุณ (GNN ใน env, action 200 ลิงก์):**
```bash
# ประเมินโมเดล 100k steps ที่เทรนแล้ว
python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
    --base-model vm_originals/SVs2/ppo_gnn_sdn_model --eval-only --eval-episodes 5

# Fine-tune ต่อ (resume) อีก 3,000 steps
python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
    --base-model vm_originals/SVs2/ppo_gnn_sdn_model --lr 1e-4 \
    --total-timesteps 3000 --tag resume
```

**แบบ B — โมเดลใหม่ (policy-side GNN, 50 ลิงก์) ผ่าน transfer learning:**
```bash
python fine_tune_sdn_agent.py --env onos --arch gnn --vm1-ip 192.168.10.165 \
    --base-model results/ppo_gnn_fast_base --freeze --lr 1e-4 \
    --total-timesteps 3000 --tag onos
```

- `--obs-mode gnn` = observation ผ่าน GNN ใน env (โมเดลเดิม), `raw` = policy รัน GNN เอง
- `--freeze` แช่แข็ง GNN extractor (เทรนเฉพาะ policy head)
- `--base-model` โหลด weights จาก pretrain (transfer learning)
- ถ้าไม่มี metrics server ที่ port 9999 ให้แก้ `use_real_metrics=False` ใน `custom_sdn_env.py`
  (จะใช้ค่า mock แทน) หรือตั้ง `step_delay=0` เพื่อให้เร็วขึ้น

### 3) Benchmark + กราฟ

```bash
python benchmark_compare.py --model-vanilla results/ppo_mlp_fast_vanilla \
                            --model-ppognn results/ppo_gnn_fast_ppognn --episodes 20
# ไม่มีโมเดล → ให้เทรนสั้น ๆ ให้เอง
python benchmark_compare.py --quick-train-steps 4000 --episodes 20
# Zero-shot generalization (topology ใหม่ ไม่เทรนใหม่)
python benchmark_compare.py --generalize --model-vanilla ... --model-ppognn ...
# ประเมินกับ ONOS จริง
python benchmark_compare.py --env onos --vm1-ip 192.168.10.165 \
    --model-vanilla ... --model-ppognn ...
```

Output: `results/benchmark_results.csv`, `results/benchmark_results.json`, `results/benchmark_metrics.png`

### 4) เตรียมสไลด์

เปิด `presentation_results.md` → เติมตัวเลขจริงจาก benchmark → วางลง Canva

## โค้ดจริงจาก VM (`vm_originals/`)

โค้ดที่ extract ออกมาจาก VM จริงของคุณ (ผ่าน SSH/SFTP, user `ino`):

- `vm_originals/SVs1/` — **VM Mininet + ONOS** (192.168.10.165):
  - `nn_topo_advanced.py` — topology 14 สวิตช์ 3 เลเยอร์ (4→6→4, 48 ลิงก์)
  - `advanced_mesh.py` — topology ทดสอบ 6 สวิตช์
  - `metrics_server.py` — ตัววัด throughput/latency/packet loss (port 9999)
- `vm_originals/SVs2/` — **VM AI brain** (192.168.10.167):
  - `sdn_network_env.py` — env ต่อ ONOS จริง (ต้นฉบับ)
  - `train_sdn_ai.py` — สคริปต์เทรน 100k steps + resume
  - `ppo_gnn_sdn_model.zip` — โมเดลเทรนเสร็จ (341 KB)
  - `checkpoints/`, `sdn-brain-checkpoints/` — checkpoint 10k–100k steps
  - `tensorboard/` — logs สำหรับดู convergence curve

## หมายเหตุทางเทคนิค

- **Observation (raw):** `node_feat(14) + edge_attr(50×2)` = 114 มิติ
  (edge feature = [utilization, normalized weight]) — layout เดียวกันทั้ง fast และ onos
  จึงโอนถ่ายโมเดลข้าม env ได้
- **Action:** น้ำหนักลิงก์ 50 ค่า ช่วง [1, 100] — ส่งเป็น `annotations.cost` ผ่าน
  `POST /onos/v1/network/configuration/links/...`
- **Reward:** Network Power `throughput^1.2 / latency` + penalty packet loss (scale 1e5)
- **PPO+GNN:** `SDNGraphFeatureExtractor` = GATConv(1→16) + GATConv(16→32) + global mean pool
- **Vanilla PPO:** MLP บน observation ตรง ๆ (ไม่มี GNN) — เป็น baseline ที่ต้องพ่ายแพ้
