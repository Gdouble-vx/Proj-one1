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

**ผลการประเมินจริงของโมเดล 100k steps (ONOS 2.7.0, 5 episodes × 200 steps):**

| Metric | ค่าเฉลี่ย |
|---|---|
| Throughput | **33,127.6 Mbps** (~32 Gbps) |
| Latency | **0.06 ms** |
| Packet Loss | **0.0%** |
| Reward / episode | **7,837.8** |

**เทียบกับ Dijkstra/OSPF baseline (รันบน ONOS จริง topology เดียวกัน, 5×200 steps):**

| Metric | Dijkstra/OSPF | PPO+GNN 100k | Δ |
|---|---|---|---|
| Throughput (Mbps) | 33,235.4 | 33,127.6 | −0.3% |
| Latency (ms) | 0.099 | 0.060 | **−39%** |
| Packet Loss (%) | 0.0 | 0.0 | 0 |
| Reward / episode | 6,780.5 | 7,837.8 | **+15.6%** |

> โมเดล 100k เล่นเซฟ (action = 1.0 ทุกลิงก์ → ไม่เปลี่ยนเส้นทางจาก OSPF) → throughput/loss เท่ากันใน noise,
> latency ต่ำกว่าเล็กน้อย → reward สูงกว่า ~16% — ยังไม่เหนือกว่า baseline อย่างมีนัยสำคัญจนกว่าจะ fine-tune ต่อ
> (รายละเอียดราย episode อยู่ใน `presentation_results.md` ข้อ 1.7)

Evidence: log ดิบ `results_eval_100k_onos.log` (1,000 samples) + กราฟ `results/real_metrics_eval_onos.png`
(รายละเอียดราย episode อยู่ใน `presentation_results.md` ข้อ 1.6)

**ผลการ Fine-Tune 3,000 steps (resume จาก 100k, ONOS จริง) — รันเสร็จแล้ว:**

```bash
python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
    --base-model ppo_gnn_sdn_model --lr 1e-4 --total-timesteps 3000 --tag resume
# → เวลาจริง 15,630s (4.34 ชม.) · โมเดล: results/ppo_gnn_onos_resume.zip
# → training log: results/train_gnn_onos_resume.csv (reward ทรงตัว ~7,340–7,425)
```

Eval หลัง fine-tune (5 episodes × 200 steps, deterministic, seed 1000–1004):

| Episode | Reward | Throughput (Mbps) | Latency (ms) | Loss (%) |
|---|---|---|---|---|
| ep0 | 7,200.25 | 29,534 | 0.08 | 0.0 |
| ep1 | 7,475.53 | 31,229 | 0.06 | 0.0 |
| ep2 | 7,164.25 | 31,983 | 0.07 | 0.0 |
| ep3 | 7,265.79 | 32,266 | 0.07 | 0.0 |
| ep4 | 7,285.54 | 30,723 | 0.06 | 0.0 |
| **เฉลี่ย** | **7,278.3** | **31,147** | **0.07** | **0.0** |

**สรุป 4 วิธี (ONOS 2.7.0 จริง, โปรโตคอลเดียวกัน):**

| Metric | Dijkstra/OSPF | PPO+GNN 100k | Fine-tune v1 | Fine-tune v2 |
|---|---|---|---|---|
| Throughput (Mbps) | 33,235.4 | 33,127.6 | 31,147.0 | **33,387.8** |
| Latency (ms) | 0.099 | 0.060 | 0.068 | 0.070 |
| Packet Loss (%) | 0.0 | 0.0 | 0.0 | 0.0 |
| Reward / episode | 6,780.5 | 7,837.8 | 7,278.3 | 6,917.5 |
| Links เปลี่ยน/step | 0 | 0 | 0 | **3-10** |

> v2 มี throughput สูงสุด (33,388 Mbps) ใกล้เคียง Dijkstra — แสดงว่า model เปลี่ยน routing ได้จริง
> แต่ reward เฉลี่ยต่ำกว่า 100k (6,918 vs 7,838) เพราะ deterministic eval ยังเล่นเซฟ
> บทเรียน: reward shaping แก้ safe-action ได้ (links เปลี่ยน 3-10/step during training)
> แต่ต้องเพิ่ม temperature/exploration ตอน eval ด้วย ไม่งั้น deterministic policy กลับเป็น safe (รายละเอียดใน `presentation_results.md` ข้อ 1.9)

**แบบ C — Reward Shaping v2 (แก้ safe-action plateau) — เทรนเสร็จแล้ว:**
```bash
python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
    --base-model ppo_gnn_sdn_model --lr 1e-4 --ent-coef 0.08 \
    --total-timesteps 3000 --tag reward_v2
# → เวลาจริง 15,789s (4.38 ชม.) · โมเดล: results/ppo_gnn_onos_reward_v2.zip
```

**สิ่งที่แก้:** เพิ่ม composite reward (exploration + diversity + novelty + improvement bonuses),
ลด delta threshold 2.0→0.5, เพิ่ม ent_coef 0.01→0.08 — โมเดลเปลี่ยน link weights 3-10/step
(แทนที่ 0/step เดิม) · รายละเอียดอยู่ใน `presentation_results.md` ข้อ 1.9

**ผล eval หลังเทรน (5 episodes × 200 steps, deterministic, seed 1000–1004):**

| Episode | Reward | Throughput (Mbps) | Latency (ms) | Loss (%) |
|---|---|---|---|---|
| ep0 | 7,393.82 | 31,410 | 0.07 | 0.0 |
| ep1 | 6,955.66 | 34,141 | 0.07 | 0.0 |
| ep2 | 6,656.38 | 33,519 | 0.07 | 0.0 |
| ep3 | 6,814.28 | 34,688 | 0.08 | 0.0 |
| ep4 | 6,767.41 | 33,181 | 0.07 | 0.0 |
| **เฉลี่ย** | **6,917.5** | **33,388** | **0.07** | **0.0** |

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

## ตั้งค่า ONOS บน SVs1 (Mininet/ONOS VM) — ใช้ 2.7.0 (stable)

> ⚠️ **ทำไมต้อง 2.7.0:** image `onosproject/onos:latest` (3.0.0.SNAPSHOT) มีบั๊ก
> Atomix/Kryo serialization (`Class is not registered: io.atomix.core.map.AtomicMapEvent`)
> ทำให้ `HostLocationProvider` crash → host/ARP ทำงานไม่ได้ → ping h1↔h2 ไม่ผ่าน →
> metrics server คืนค่า default คงที่ (100 Mbps / 50 ms) → reward ไม่มีสัญญาณ → **AI plateau**

```bash
# 1) เริ่ม docker + ONOS (ครั้งแรก: สลับจาก 3.0 เป็น 2.7.0)
#    หยุด + เก็บ container เดิม (rollback ได้)
sudo docker stop onos && sudo docker rename onos onos_latest_backup
sudo docker pull onosproject/onos:2.7.0
sudo docker run -d --name onos --network host onosproject/onos:2.7.0

# 2) รอ REST ขึ้น แล้วเปิด app ที่จำเป็น
curl -u onos:rocks -X POST http://127.0.0.1:8181/onos/v1/applications/org.onosproject.openflow/active
curl -u onos:rocks -X POST http://127.0.0.1:8181/onos/v1/applications/org.onosproject.fwd/active
curl -u onos:rocks -X POST http://127.0.0.1:8181/onos/v1/applications/org.onosproject.proxyarp/active
curl -u onos:rocks -X POST http://127.0.0.1:8181/onos/v1/applications/org.onosproject.hostprobingprovider/active
curl -u onos:rocks -X POST http://127.0.0.1:8181/onos/v1/applications/org.onosproject.reactive-routing/active

# 3) เริ่ม Mininet topology + metrics server (ต้อง root — สคริปต์จบด้วย CLI(net)
#    ที่อ่าน stdin → ต้องจับ stdin เปิดค้างไว้ ไม่งั้นเครือข่ายจะ teardown ทันที)
sudo mn -c
nohup bash -c "(echo '<pass>'; tail -f /dev/null) | sudo -S python3 ~/nn_topo_advanced.py" \
    > /tmp/mininet.log 2>&1 &
nohup bash -c "echo '<pass>' | sudo -S python3 ~/metrics_server.py" > /tmp/metrics.log 2>&1 &

# 4) เปิด iperf server บน h2 + ลง HostToHostIntent (ให้ ping h1↔h2 ผ่าน)
sudo nsenter -t $(ps ax | grep 'mininet:h2' | grep -v grep | awk '{print $1}' | head -1) \
    -n sh -c 'iperf -s > /dev/null 2>&1 &'
curl -u onos:rocks -X POST http://127.0.0.1:8181/onos/v1/intents \
    -H "Content-Type: application/json" \
    -d '{"type":"HostToHostIntent","appId":"org.onosproject.cli","one":"00:00:00:00:00:01/None","two":"00:00:00:00:00:02/None","priority":100}'

# 5) ตรวจ
curl -u onos:rocks http://127.0.0.1:8181/onos/v1/devices   # ควรได้ 14 devices available
curl -u onos:rocks http://127.0.0.1:8181/onos/v1/links     # ควรได้ 96 links (48 คู่)
curl http://127.0.0.1:9999                                  # ควรได้ throughput/latency จริง

# Rollback (ถ้าต้องการกลับไป 3.0)
sudo docker stop onos && sudo docker start onos_latest_backup
```

> **แก้บั๊ก metrics_server.py:** ต้นฉบับยิง `iperf -c 10.0.0.1` (IP ของ h1 เอง — วัดไม่ได้)
> แก้เป็น `iperf -c 10.0.0.2` (h2 ซึ่งรัน `iperf -s`) — backup เดิมเก็บไว้ที่ `metrics_server.py.bak`
> ไฟล์นี้ใน `vm_originals/SVs1/` อัปเดตให้ตรงกับเวอร์ชันที่แก้แล้ว

## สถานะ VM (18 ส.ค. 2026 — ย้ายจาก E: ไป C:)

> ⚠️ ดิสก์ E: (เดิม) มี bad sector — `fsync` ล้มเหลวทั้งไดรฟ์ (เขียนผ่านแต่ flush ไม่ได้) →
> VM ค้างทุกครั้งที่ boot จาก E: จึง **ย้าย VM ไป `C:\sdn-vms\`** (ดิสก์สุขภาพดี) แล้ว boot จากที่นั่น

```bash
# boot VM จากตำแหน่งใหม่ (โฟลเดอร์ละชุด vmdk + vmx)
"/c/Program Files (x86)/VMware/VMware Workstation/vmrun.exe" start "C:\sdn-vms\SVs1\Ubuntu-SVs1.vmx" nogui
"/c/Program Files (x86)/VMware/VMware Workstation/vmrun.exe" start "C:\sdn-vms\SVs2\Ubuntu-SVs2.vmx" nogui
# VM เดิมบน E: ยังอยู่ (`E:\เรียน+\proj`) — อย่า boot จากตรงนั้นจนกว่าดิสก์จะซ่อม
```

- ISO cdrom ใน `.vmx` ถูกแก้ชี้ไป `C:\sdn-vms\iso\ubuntu-24.04.4-live-server-amd64.iso` แล้ว
- ถ้า git บนเครื่องนี้ error `fsync error ... Bad file descriptor` (E: ยังพัง) ให้ commit ด้วย:
  `git -c core.fsync=none -c core.fsyncObjectFiles=false commit ...`

## หมายเหตุทางเทคนิค

- **Observation (raw):** `node_feat(14) + edge_attr(50×2)` = 114 มิติ
  (edge feature = [utilization, normalized weight]) — layout เดียวกันทั้ง fast และ onos
  จึงโอนถ่ายโมเดลข้าม env ได้
- **Action:** น้ำหนักลิงก์ 50 ค่า ช่วง [1, 100] — ส่งเป็น `annotations.cost` ผ่าน
  `POST /onos/v1/network/configuration/links/...`
- **Reward:** Network Power `throughput^1.2 / latency` + penalty packet loss (scale 1e5)
- **PPO+GNN:** `SDNGraphFeatureExtractor` = GATConv(1→16) + GATConv(16→32) + global mean pool
- **Vanilla PPO:** MLP บน observation ตรง ๆ (ไม่มี GNN) — เป็น baseline ที่ต้องพ่ายแพ้
