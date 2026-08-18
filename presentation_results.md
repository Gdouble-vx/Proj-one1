# 📊 ข้อมูลนำเสนอผลงาน (Presentation Results) — SDN AI Agent

> ไฟล์นี้สรุปผลการรัน **Fine-Tuning (PPO+GNN)** และ **Benchmark** ให้อยู่ในรูปแบบตาราง + ข้อความ
> พร้อมนำไปวางในสไลด์ Canva ได้เลย
>
> ✅ ค่าที่เป็น **ตัวเลขจริงจาก Fast Simulator**: Dijkstra / ECMP
> ✅ ค่าที่เป็น **ตัวเลขจริงจาก ONOS (REST)**: PPO+GNN 100k steps — ดูข้อ 1.6
> ⚠️ ค่าที่เป็น **placeholder `[____]`**: ต้องรัน `benchmark_compare.py` บนเครื่อง ML ของคุณเพื่อเติมตัวเลข PPO ที่เหลือ

---

## 1️⃣ ตาราง Benchmark หลัก (ใช้ตอบอาจารย์)

**การทดลอง:** 14 nodes / 50 links, 12 flows สุ่มใหม่ทุก episode (demand 100–400 Mbps), link capacity 500 Mbps,
ประเมิน 20 episodes บน scenario เดียวกันทุก method (fix seed)

| Method | Avg Throughput (Mbps) | Avg Latency (ms) | Packet Loss (%) | Avg Reward |
|---|---|---|---|---|
| **Dijkstra / OSPF** (baseline) | 2759.4 | 8.01 | 9.75 | 0.0208 |
| **ECMP** (baseline) | 2991.0 | 1.59 | 2.36 | 0.0982 |
| **Vanilla PPO (MLP)** | `[____]` | `[____]` | `[____]` | `[____]` |
| **PPO + GNN (Proposed)** ⭐ | `[____]` | `[____]` | `[____]` | `[____]` |

**วิธีเติมตัวเลข:** รัน

```bash
python fine_tune_sdn_agent.py --env fast --arch gnn --total-timesteps 10000 --tag ppognn
python fine_tune_sdn_agent.py --env fast --arch mlp --total-timesteps 10000 --tag vanilla
python benchmark_compare.py --model-vanilla results/ppo_mlp_fast_vanilla \
                            --model-ppognn results/ppo_gnn_fast_ppognn --episodes 20
```

ผลลัพธ์อัตโนมัติ: `results/benchmark_results.csv`, `results/benchmark_results.json`, `results/benchmark_metrics.png`

---

## 1.5️⃣ โมเดลจริงของคุณ (เทรนบน ONOS จริง 100,000 steps)

Extract ออกจาก VM แล้วอยู่ใน `vm_originals/SVs2/`:

| Artifact | รายละเอียด |
|---|---|
| `ppo_gnn_sdn_model.zip` | โมเดล PPO+GNN เทรนเสร็จ (GNN ใน env, obs 32 มิติ, action 200 ลิงก์) |
| `checkpoints/` | checkpoint ทุก 10k steps (10k → 100k) — ใช้ดู convergence ได้ |
| `tensorboard/` | logs สำหรับวาดกราฟ reward curve |
| `sdn_network_env.py` + `train_sdn_ai.py` | โค้ดต้นฉบับของคุณ |

**ประเมินโมเดลจริงกับ ONOS (ต้องเปิด VM + Mininet + metrics server ก่อน):**

```bash
python fine_tune_sdn_agent.py --env onos --obs-mode gnn --num-links 200 \
    --base-model vm_originals/SVs2/ppo_gnn_sdn_model --eval-only --eval-episodes 5
```

> เติมตัวเลขที่ได้ (throughput/latency/loss) ลงในตารางข้อ 1 แทนช่อง PPO+GNN — นี่คือผลการรันจริงของคุณ

## 1.6️⃣ ผลการประเมินจริง: โมเดล 100k steps บน ONOS 2.7.0 ✅ (eval-only, 5 episodes)

**รันเมื่อ:** 18 ส.ค. 2026 — `--env onos --obs-mode gnn --num-links 200 --base-model ppo_gnn_sdn_model --eval-only --eval-episodes 5`
(200 steps/episode, deterministic policy, seed 1000–1004, วัด metric จริงผ่าน REST :9999)

| Episode | Reward | Avg Throughput (Mbps) | Avg Latency (ms) | Packet Loss (%) |
|---|---|---|---|---|
| ep0 | 7400.01 | 30,856 | 0.07 | 0.0 |
| ep1 | 7831.25 | 33,504 | 0.07 | 0.0 |
| ep2 | 9712.93 | 34,038 | 0.05 | 0.0 |
| ep3 | 7333.12 | 34,794 | 0.04 | 0.0 |
| ep4 | 6911.85 | 32,446 | 0.08 | 0.0 |
| **เฉลี่ย (5 episodes)** | **7837.83** | **33,127.6** | **0.06** | **0.0** |
| เฉลี่ยทุก step (1,000 samples) | — | 31,972 | 0.085 | 0.0 |

**ข้อสังเกตสำคัญ:**
- พฤติกรรมของโมเดล 100k = **เล่นเซฟ**: ออก action = 1.0 ทั้ง 200 ลิงก์ (น้ำหนักต่ำสุด) → ไม่มีลิงก์ไหนถูกเปลี่ยน
  (log แสดง "ข้าม 48 ที่ไม่เปลี่ยน" ทุก step) → throughput/latency คงที่รอบ ๆ ค่า default network
- ตัวเลขนี้คือ **ค่า real data path บน OVS software switch** (topology 14 สวิตช์ / 48 ลิงก์) — ต่าง scale
  จาก fast simulator (500 Mbps/link) → **ห้ามเทียบข้ามตารางกับข้อ 1 ตรง ๆ** ใช้เป็นหลักฐานว่า
  "ระบบวัดผลจริงได้ และโมเดลทำงานบน ONOS จริง" มากกว่า
- Evidence: log ดิบ `results_eval_100k_onos.log` (1,000 samples) + กราฟ `results/real_metrics_eval_onos.png`

## 1.7️⃣ เปรียบเทียบ Dijkstra/OSPF vs PPO+GNN 100k (ONOS จริง) ✅

**การทดลอง:** รันทั้งสองวิธีบน topology เดียวกัน (14 สวิตช์ / 48 ลิงก์, ONOS 2.7.0, Mininet OVS) ด้วยโปรโตคอลเดียวกับข้อ 1.6
(5 episodes × 200 steps, seeds 1000–1004, deterministic)

- **Dijkstra/OSPF baseline** = ไม่แตะน้ำหนักลิงก์ (annotations ว่าง = cost default = OSPF/equal-cost) — โปรโตคอลเดียวกับที่ PPO eval เริ่มต้น
- **PPO+GNN 100k** = โมเดล `ppo_gnn_sdn_model` — ดูข้อ 1.6

| Episode | Dijkstra — Throughput (Mbps) | Dijkstra — Latency (ms) | Dijkstra — Reward | PPO — Reward |
|---|---|---|---|---|
| ep0 | 34,968 | 0.14 | 6,915 | 7,400.01 |
| ep1 | 33,956 | 0.08 | 6,811 | 7,831.25 |
| ep2 | 30,758 | 0.08 | 6,823 | 9,712.93 |
| ep3 | 34,889 | 0.10 | 6,539 | 7,333.12 |
| ep4 | 31,606 | 0.10 | 6,814 | 6,911.85 |
| **เฉลี่ย** | **33,235.4** | **0.099** | **6,780.5** | **7,837.8** |

| Metric | Dijkstra/OSPF | PPO+GNN 100k | Δ |
|---|---|---|---|
| Avg Throughput (Mbps) | 33,235.4 | 33,127.6 | −0.3% |
| Avg Latency (ms) | 0.099 | 0.060 | **−39%** |
| Packet Loss (%) | 0.0 | 0.0 | 0 |
| Avg Reward | 6,780.5 | 7,837.8 | **+15.6%** |

**การตีความ (ซื่อตรง — ควรนำเสนอตามนี้):**
- ทั้งคู่วัดบน OSPF state เดียวกัน; โมเดล 100k **เล่นเซฟ** (action = 1.0 ทุกลิงก์) → ไม่ได้เปลี่ยนเส้นทางจาก baseline จริง
  → throughput/loss แทบเท่ากัน (ต่างกันใน noise ของการวัด)
- latency ต่ำกว่าเล็กน้อย (0.06 vs 0.10 ms) → reward สูงกว่า ~16% แต่ยังอยู่ในช่วงความแปรปรวน
- **ข้อสรุปที่นำเสนอได้:** โมเดล 100k ยังไม่เหนือกว่า baseline อย่างมีนัยสำคัญ — คาดว่าจะได้ผลต่างจริง
  หลัง fine-tune ต่อบน ONOS 2.7.0 (metric จริงมีสัญญาณแล้ว) — นี่คือ motivation ของขั้น fine-tune 3,000 steps

Evidence: `results_dijkstra_onos.json` (บน VM ก่อน disk fail — ตัวเลขยืนยันในตารางนี้)

## 2️⃣ Zero-Shot Generalization Test (โจทย์ "ย้าย Topology โดยไม่เทรนใหม่")

| Method | Topology เดิม — Packet Loss (%) | Topology ใหม่ (Zero-Shot) — Packet Loss (%) |
|---|---|---|
| Vanilla PPO (MLP) | `[____]` | `[____]` |
| PPO + GNN (Proposed) ⭐ | `[____]` | `[____]` |

```bash
python benchmark_compare.py --generalize --model-vanilla results/ppo_mlp_fast_vanilla \
                            --model-ppognn results/ppo_gnn_fast_ppognn --episodes 20
```

> คาดหวัง: GNN extractor อ่านโครงสร้างกราฟ (edge_index ถูกสลับเป็น topology ใหม่โดยไม่เทรนใหม่)
> จึงควร generalize ได้ดีกว่า MLP ที่จำ feature vector แบน ๆ

---

## 3️⃣ ตาราง Fine-Tuning / Transfer Learning (ตอบโจทย์ "เวลาพอไหม")

| ขั้นตอน | จำนวน Steps | เวลาจริง | หมายเหตุ |
|---|---|---|---|
| เทรนจากศูนย์บน ONOS/Mininet (เดิม) | 100,000 | **~29.2 ชม.** | ~1 FPS ผ่าน REST API |
| Pretrain บน FastSDNEnv (simulator) | 10,000 | `[____]` นาที | หลายพัน step/วินาที |
| **Fine-Tune บน ONOS จริง** (freeze GNN + lr=1e-4) | 2,000–5,000 | **~15–30 นาที** (เป้าหมาย) | โหลด base weights → ต่อยอด |

คำสั่ง:

```bash
# pretrain บน simulator
python fine_tune_sdn_agent.py --env fast --arch gnn --total-timesteps 10000 --tag base

# fine-tune กับ ONOS จริง (transfer learning + freeze GNN extractor)
python fine_tune_sdn_agent.py --env onos --arch gnn --vm1-ip 192.168.10.165 \
    --base-model results/ppo_gnn_fast_base --freeze --lr 1e-4 --total-timesteps 3000 --tag onos
```

Training log อัตโนมัติ: `results/train_gnn_fast_base.csv` (มี column `fps` ใช้อ้างอิงความเร็วได้)

---

## 4️⃣ ข้อความสำเร็จรูปสำหรับสไลด์ Canva

### Slide: ปัญหา (Problem)
> "การเทรน PPO จากศูนย์บน SDN จริง (Mininet + ONOS ผ่าน REST API) ใช้เวลา 29.2 ชั่วโมงสำหรับ 100,000 steps
> หรือ ~1 step/วินาที — ไม่ทันกำหนดส่งงาน และโมเดลติด plateau เพราะ penalty สูงจน agent เลือกเล่นปลอดภัย"

### Slide: กลยุทธ์ (Solution)
> "ใช้ Transfer Learning: pretrain บน FastSDNEnv (simulator ความเร็วสูง) → freeze ชั้น GNN feature extractor
> → fine-tune เฉพาะ policy head ด้วย learning rate 1e-4 บน ONOS จริงเพียง 2,000–5,000 steps
> ลดเวลาจาก 29.2 ชั่วโมง เหลือ 15–30 นาที"

### Slide: วิธีวัดผล (Methodology)
> "เปรียบเทียบ 3 กลุ่ม: (1) ดั้งเดิม — Dijkstra/OSPF และ ECMP (2) DRL มาตรฐาน — Vanilla PPO + MLP
> (3) วิธีที่นำเสนอ — PPO + GNN (GATConv 2 ชั้น + global mean pooling) โดยวัด Throughput, Latency,
> Packet Loss, และทดสอบ Zero-Shot Generalization บน topology ใหม่โดยไม่เทรนใหม่"

### Slide: สรุปผล (Key Findings) — ตัวเลขจาก simulator (ตัวอย่าง)
> "บน fast simulator 14 โหนด/50 ลิงก์: Dijkstra มี packet loss 9.75% และ latency 8.01 ms
> ในขณะที่ ECMP ลด loss เหลือ 2.36% (latency 1.59 ms) — แสดงให้เห็นว่าการเลือกเส้นทางแบบ load-aware
> สำคัญต่อเครือข่ายหนาแน่น และเป็นจุดที่ PPO+GNN ควรต่อยอดให้ดีขึ้นได้อีก"

### Slide: สถาปัตยกรรม (Architecture)
> "GNN (GATConv + global mean pool) ใช้บีบอัดสถานะ topology (14 โหนด / 50 ลิงก์) เป็น state vector
> 32 มิติ → policy head (MLP 64-64) ออก action = ค่าน้ำหนักลิงก์ 50 ค่า → ส่งไปปรับผ่าน ONOS REST API
> → วัดผลตอบแทนจาก Network Power (throughput^1.2 / latency) พร้อม penalty เรื่อง packet loss"

---

## 5️⃣ Checklist ก่อนนำเสนอ

- [ ] รัน fine-tune ครบ (pretrain fast → fine-tune onos) → ได้ `results/ppo_gnn_fast_*.zip` + training log
- [ ] รัน `benchmark_compare.py` → เติมตัวเลข PPO ในตารางข้อ 1
- [ ] รัน `--generalize` → เติมตารางข้อ 2
- [ ] เอา `results/benchmark_metrics.png` ไปแปะในสไลด์ (หรือทำกราฟใหม่ใน Canva)
- [ ] ถ่าย screenshot หน้าจอ ONOS GUI + Mininet CLI ประกอบสไลด์
- [ ] เตรียมข้อมูลสถาปัตยกรรม + flow diagram (ดู README.md)
