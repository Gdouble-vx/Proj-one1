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
| `results/` | output: โมเดล `.zip`, training log `.csv`, benchmark results + charts |

> หมายเหตุ: ไฟล์เดิม `import gymnasium as gym.py` ถูก refactor ไปอยู่ใน `custom_sdn_env.py`
> (เพิ่ม `obs_mode="raw"`, `max_links=50`, dedupe ลิงก์สองทิศทาง) — ลบไฟล์เดิมได้ถ้าต้องการ

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

```bash
python fine_tune_sdn_agent.py --env onos --arch gnn --vm1-ip 192.168.10.165 \
    --base-model results/ppo_gnn_fast_base --freeze --lr 1e-4 \
    --total-timesteps 3000 --tag onos
```

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

## หมายเหตุทางเทคนิค

- **Observation (raw):** `node_feat(14) + edge_attr(50×2)` = 114 มิติ
  (edge feature = [utilization, normalized weight]) — layout เดียวกันทั้ง fast และ onos
  จึงโอนถ่ายโมเดลข้าม env ได้
- **Action:** น้ำหนักลิงก์ 50 ค่า ช่วง [1, 100] — ส่งเป็น `annotations.cost` ผ่าน
  `POST /onos/v1/network/configuration/links/...`
- **Reward:** Network Power `throughput^1.2 / latency` + penalty packet loss (scale 1e5)
- **PPO+GNN:** `SDNGraphFeatureExtractor` = GATConv(1→16) + GATConv(16→32) + global mean pool
- **Vanilla PPO:** MLP บน observation ตรง ๆ (ไม่มี GNN) — เป็น baseline ที่ต้องพ่ายแพ้
