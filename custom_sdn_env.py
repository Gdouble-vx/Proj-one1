"""
custom_sdn_env.py — CustomSDNEnv: เชื่อมต่อกับ ONOS Controller จริงผ่าน REST API

ย้ายมาจากไฟล์เดิม ("import gymnasium as gym.py") และเพิ่ม:
  - obs_mode="raw"  : คืน observation แบบ raw graph state (node + edge features)
                      ให้ policy เป็นฝ่ายรัน GNN (ใช้กับ fine_tune_sdn_agent.py --env onos)
  - obs_mode="gnn"  : คืน observation ที่ผ่าน GNN encoder ใน env แล้ว (พฤติกรรมเดิม)
  - max_links       : 50 (ตรงกับ FastSDNEnv) → โอนถ่ายน้ำหนักระหว่าง fast/onos ได้
  - dedupe ลิงก์สองทิศทางใน raw mode (ONOS คืน link มา 2 ทิศทาง)

ข้อควรรู้:
  - REST API ของ ONOS: http://<vm1_ip>:8181/onos/v1/...  auth=onos/rocks
  - ตั้งค่า link weight: POST /onos/v1/network/configuration/links/<src>/<sport>-<dst>/<dport>
  - วัด metrics จริง: ผ่าน server ที่ http://<vm1_ip>:9999 (ใช้ use_real_metrics=False ได้ถ้าไม่มี)
"""

import time

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from requests.auth import HTTPBasicAuth
import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, global_mean_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None
    GATConv = None
    global_mean_pool = None


# ----------------------------------------------------------------------------
# GNN Encoder (ฝังอยู่ใน env) — ใช้กับ obs_mode="gnn"
# ----------------------------------------------------------------------------
class NetworkGNNEncoder(torch.nn.Module):
    def __init__(self, node_features, edge_features, output_dim):
        super(NetworkGNNEncoder, self).__init__()
        self.conv1 = GATConv(node_features, 16, edge_dim=edge_features)
        self.conv2 = GATConv(16, output_dim, edge_dim=edge_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        state_vector = global_mean_pool(x, batch)
        return state_vector


# ----------------------------------------------------------------------------
# Reward & Metric helpers
# ----------------------------------------------------------------------------
def calculate_reward(throughput, latency, packet_loss, alpha=1.2):
    """Reward เดิม — metric-based only"""
    if latency <= 0:
        latency = 0.001
    network_power = (throughput ** alpha) / latency
    penalty = 0.0
    if packet_loss > 0.05:
        penalty = -100.0
    elif packet_loss > 0.01:
        penalty = -20.0
    elif packet_loss > 0.001:
        penalty = -5.0
    reward = network_power + penalty
    return float(reward)


# ค่าคงที่สำหรับ reward shaping
_EXPLORATION_K = 0.5       # กระตุ้นให้ model เปลี่ยน link weights จาก step ก่อนหน้า
_DIVERSITY_K = 0.3         # กระตุ้นให้ action ไม่ uniform (คิดว่า link ไหนควรเปลี่ยน)
_IMPROVEMENT_K = 0.5       # รางวัลเมื่อ metric ดีขึ้น
_NOVELTY_K = 0.05          # รางวัลเมื่อ action ห่างจาก baseline (all=1.0) — ทํางานตั้งแต่ step 1
_STAGNATION_PENALTY = -0.5 # ลงโทษเมื่อ action ทั้งหมดเท่ากัน (uniform) → บังคับให้คิด
_consecutive_uniform = 0    # นับจำนวน step ที่ action ทั้งหมดเท่ากัน


def compute_exploration_bonus(action, prev_action):
    """โบนัสเมื่อ model ลองเปลี่ยน link weights จาก step ก่อนหน้า
    ถ้า action == prev_action → bonus = 0 (ไม่เปลี่ยนอะไร)
    ถ้าเปลี่ยนเยอะ → bonus สูง (กระตุ้นให้ลองสิ่งใหม่)"""
    global _consecutive_uniform
    if prev_action is None:
        return 0.0
    delta = np.abs(action - prev_action)
    bonus = float(np.mean(delta)) * _EXPLORATION_K
    # ถ้าไม่เปลี่ยนอะไรเลย เพิ่มจำนวน consecutive uniform
    if float(np.max(delta)) < 0.01:
        _consecutive_uniform += 1
    else:
        _consecutive_uniform = 0
    return bonus


def compute_diversity_bonus(action):
    """โบนัสเมื่อ action ไม่ uniform — ถ้า model ให้ค่าต่างกัน across links
    แสดงว่ากำลัง "คิด" ว่า link ไหนควรเปลี่ยน vs ไม่เปลี่ยน
    + ลงโทษถ้า action uniform (กระตุ้นให้คิดต่าง)"""
    std_val = float(np.std(action))
    diversity = std_val * _DIVERSITY_K
    # ลงโทษถ้า action ทั้งหมดเท่ากัน (ไม่คิดเลย)
    if std_val < 0.01:
        diversity += _STAGNATION_PENALTY
    return diversity


def compute_novelty_bonus(action):
    """โบนัสเมื่อ action ห่างจาก baseline (all=1.0 = OSPF default)
    ทํางานตั้งแต่ step 1 — แก้ปัญหา chicken-and-egg"""
    baseline = np.ones_like(action, dtype=np.float32)
    novelty = float(np.mean(np.abs(action - baseline)))
    return novelty * _NOVELTY_K


def compute_improvement_bonus(current_metrics, prev_metrics):
    """โบนัสเมื่อ metric ดีขึ้นจาก step ก่อนหน้า"""
    if prev_metrics is None:
        return 0.0
    curr_t = current_metrics.get("throughput", 0)
    prev_t = prev_metrics.get("throughput", 0)
    curr_l = current_metrics.get("latency", 100)
    prev_l = prev_metrics.get("latency", 100)
    bonus = 0.0
    # throughput ดีขึ้น
    if prev_t > 0 and curr_t > prev_t:
        bonus += _IMPROVEMENT_K * ((curr_t - prev_t) / prev_t)
    # latency ลดลง
    if prev_l > 0 and curr_l < prev_l:
        bonus += _IMPROVEMENT_K * ((prev_l - curr_l) / prev_l)
    return float(bonus)


def measure_real_metrics(vm1_ip="192.168.10.165"):
    """ดึง throughput/latency/packet_loss จาก metrics server (http://<ip>:9999)."""
    try:
        response = requests.get(f"http://{vm1_ip}:9999", timeout=30)
        data = response.json()
        return data["throughput"], data["latency"], data["packet_loss"]
    except Exception as e:
        print(f"[Metric] error: {e}")
        return 100.0, 50.0, 0.0


# ----------------------------------------------------------------------------
# ONOS Environment
# ----------------------------------------------------------------------------
class CustomSDNEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, vm1_ip="192.168.10.165", num_nodes=14, max_links=50,
                 gnn_output_dim=32, obs_mode="raw", use_real_metrics=True,
                 step_delay=2.5, seed=42):
        """obs_mode: "raw" = node+edge features ตรงกับ FastSDNEnv (สำหรับ policy-side GNN)
                      "gnn" = observation ผ่าน GNN encoder ใน env (พฤติกรรมเดิม)
        step_delay: ค่าเริ่มต้น 2.5s ตามของจริง (เป็นสาเหตุหลักของ ~1 FPS) — ตั้ง 0 ได้ถ้าต้องการเร็ว"""
        super(CustomSDNEnv, self).__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric ไม่พร้อมใช้งาน — จำเป็นสำหรับ CustomSDNEnv")

        self.vm1_ip = vm1_ip
        self.num_nodes = num_nodes
        self.max_links = max_links          # ตรงกับ FastSDNEnv เพื่อให้ transfer ได้
        self.auth = HTTPBasicAuth("onos", "rocks")
        self.obs_mode = obs_mode
        self.use_real_metrics = use_real_metrics  # ← เปลี่ยนเป็น False ถ้า SSH/metrics server ไม่ได้
        self.step_delay = step_delay
        self.seed = seed
        self.step_count = 0
        self.prev_weights = {}              # cache ค่าน้ำหนักเก่า → POST เฉพาะลิงก์ที่เปลี่ยน
        self.prev_action = None             # action จาก step ก่อนหน้า (สำหรับ exploration bonus)
        self.prev_metrics = None            # metrics จาก step ก่อนหน้า (สำหรับ improvement bonus)

        # observation: node_feat(num_nodes) + edge_attr(max_links*2) เมื่อ raw
        if obs_mode == "raw":
            self.observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(num_nodes + 2 * max_links,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(gnn_output_dim,), dtype=np.float32)

        self.action_space = spaces.Box(low=1.0, high=100.0,
                                       shape=(max_links,), dtype=np.float32)

        # GNN encoder สำหรับ obs_mode="gnn"
        self.gnn_encoder = NetworkGNNEncoder(node_features=1, edge_features=2,
                                             output_dim=gnn_output_dim)
        self.gnn_encoder.eval()

        # เก็บ topology ล่าสุด (device_id -> index)
        self.device_map = {}
        self.edge_index = np.zeros((2, max_links), dtype=np.int64)  # pad self-loop (0,0)
        self.last_metrics = {"throughput": 0.0, "latency": 0.0, "packet_loss": 0.0,
                             "link_utilization": np.zeros(max_links, dtype=np.float32)}

    # ------------------------------------------------------------------ REST
    def _get_topology(self):
        """ดึง devices + links จาก ONOS REST API → (device_map, unique_links).
        unique_links: list of (src_idx, dst_idx, src_port, dst_port)
        (dedupe สองทิศทาง เพราะ ONOS คืน link มาทั้ง 2 ทิศทาง)"""
        devices_url = f"http://{self.vm1_ip}:8181/onos/v1/devices"
        links_url = f"http://{self.vm1_ip}:8181/onos/v1/links"

        try:
            devices_response = requests.get(devices_url, auth=self.auth, timeout=5)
            links_response = requests.get(links_url, auth=self.auth, timeout=5)
            if devices_response.status_code != 200 or links_response.status_code != 200:
                print(f"[ONOS] status devices={devices_response.status_code} "
                      f"links={links_response.status_code} — ใช้ topology ว่าง")
                return {}, []

            devices_data = devices_response.json()["devices"]
            links_data = links_response.json()["links"]
            device_map = {d["id"]: i for i, d in enumerate(devices_data)}

            # dedupe สองทิศทาง: {frozenset(device pair): (src_idx, dst_idx, sport, dport)}
            seen = {}
            for link in links_data:
                src_id = link["src"]["device"]
                dst_id = link["dst"]["device"]
                if src_id not in device_map or dst_id not in device_map:
                    continue
                si, di = device_map[src_id], device_map[dst_id]
                key = frozenset((si, di))
                if key not in seen:
                    seen[key] = (si, di, link["src"]["port"], link["dst"]["port"])
            unique_links = list(seen.values())
            return device_map, unique_links
        except Exception as e:
            print(f"Error fetching data from ONOS: {e}")
            return {}, []

    def _get_network_state_from_onos(self):
        """สร้าง observation จาก topology ปัจจุบัน"""
        device_map, links = self._get_topology()
        self.device_map = device_map
        n_dev = len(device_map)
        n_pad = min(n_dev, self.num_nodes)

        if self.obs_mode == "raw":
            return self._build_raw_observation(links, n_dev)
        else:
            return self._build_gnn_observation(links, n_dev)

    def _build_raw_observation(self, links, n_dev):
        """obs = node_feat(num_nodes) + edge_attr(max_links*2) — เหมือน FastSDNEnv"""
        node_feat = np.zeros(self.num_nodes, dtype=np.float32)
        edge_attr = np.zeros((self.max_links, 2), dtype=np.float32)
        edge_index = np.zeros((2, self.max_links), dtype=np.int64)

        for i, (si, di, _, _) in enumerate(links[: self.max_links]):
            edge_index[0, i] = si
            edge_index[1, i] = di
            # edge features: [utilization estimate, normalized cost]
            # (ถ้าใช้ stats endpoint จริงได้ ให้เปลี่ยนตรงนี้เป็น utilization จริง)
            edge_attr[i, 0] = 0.1
            edge_attr[i, 1] = 0.1
            if si < self.num_nodes:
                node_feat[si] = max(node_feat[si], 0.1)
            if di < self.num_nodes:
                node_feat[di] = max(node_feat[di], 0.1)

        self.edge_index = edge_index
        self.last_metrics["link_utilization"] = edge_attr[:, 0]
        return np.concatenate([node_feat, edge_attr.reshape(-1)]).astype(np.float32)

    def _build_gnn_observation(self, links, n_dev):
        src_list, dst_list, edge_features = [], [], []
        for si, di, _, _ in links:
            src_list.append(si)
            dst_list.append(di)
            edge_features.append([0.1, 2.0])

        if len(src_list) == 0:
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            edge_attr = torch.tensor([[0.1, 2.0]] * 4, dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        node_features = [[0.0] for _ in range(n_dev if n_dev > 0 else 3)]
        x = torch.tensor(node_features, dtype=torch.float)
        network_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        with torch.no_grad():
            state_vector = self.gnn_encoder(network_graph)
        return state_vector.numpy().flatten().astype(np.float32)

    # ------------------------------------------------------------------ apply
    def _apply_weights_to_onos(self, action):
        """POST ค่าน้ำหนักลิงก์ใหม่ไปที่ ONOS network configuration
        ส่งเฉพาะลิงก์ที่ weight เปลี่ยน > 0.5 หรือครั้งแรก (ลด REST calls)"""
        try:
            _, links = self._get_topology()
            device_ids = list(self.device_map.keys())
            n_changed = 0
            for idx, (si, di, src_port, dst_port) in enumerate(links[: self.max_links]):
                weight_value = float(action[idx])
                prev = self.prev_weights.get(idx)
                if prev is None or abs(weight_value - prev) > 0.5:
                    src_device = device_ids[si]
                    dst_device = device_ids[di]
                    config_url = (f"http://{self.vm1_ip}:8181/onos/v1/network/configuration/"
                                  f"links/{src_device}/{src_port}-{dst_device}/{dst_port}")
                    payload = {"annotations": {"cost": str(weight_value)}}
                    requests.post(config_url, json=payload, auth=self.auth, timeout=5)
                    self.prev_weights[idx] = weight_value
                    n_changed += 1
            print(f"[AI Action - Step {self.step_count + 1}] อัปเดต Link Weights "
                  f"{n_changed} ลิงก์ (ข้าม {len(links[: self.max_links]) - n_changed} ที่ไม่เปลี่ยน)")
        except Exception as e:
            print(f"Error sending action to ONOS: {e}")

    # ------------------------------------------------------------ gymnasium
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_weights = {}
        self.prev_action = None
        self.prev_metrics = None
        global _consecutive_uniform
        _consecutive_uniform = 0
        observation = self._get_network_state_from_onos()
        return observation, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), 1.0, 100.0)
        self._apply_weights_to_onos(action)

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        # วัดค่าจริงหรือ mock
        if self.use_real_metrics:
            throughput, latency, packet_loss = measure_real_metrics(vm1_ip=self.vm1_ip)
            print(f"[Metrics] Throughput={throughput:.1f}Mbps Latency={latency:.1f}ms "
                  f"Loss={packet_loss:.3f}")
        else:
            mean_weight = float(np.mean(action))
            throughput = float(np.clip(1000.0 - mean_weight * 2, 100, 1000))
            latency = float(np.clip(5.0 + mean_weight * 0.3, 1, 200))
            packet_loss = float(np.clip(mean_weight / 5000, 0, 0.1))

        new_metrics = {"throughput": throughput, "latency": latency,
                       "packet_loss": packet_loss}

        # === Composite Reward (5 components) ===
        # 1) Metric reward (เดิม)
        metric_reward = calculate_reward(throughput, latency, packet_loss) / 1e5
        # 2) Exploration bonus: กระตุ้นให้ model ลองเปลี่ยน link weights จาก step ก่อนหน้า
        exploration_bonus = compute_exploration_bonus(action, self.prev_action)
        # 3) Diversity bonus: กระตุ้นให้ action ไม่ uniform + ลงโทษ uniform
        diversity_bonus = compute_diversity_bonus(action)
        # 4) Novelty bonus: รางวัลเมื่อ action ห่างจาก OSPF baseline (all=1.0)
        novelty_bonus = compute_novelty_bonus(action)
        # 5) Improvement bonus: รางวัลเมื่อ metric ดีขึ้นจาก step ก่อนหน้า
        improvement_bonus = compute_improvement_bonus(new_metrics, self.prev_metrics)

        reward = metric_reward + exploration_bonus + diversity_bonus + novelty_bonus + improvement_bonus

        print(f"[Reward-Step {self.step_count + 1}] metric={metric_reward:.6f} "
              f"explore={exploration_bonus:.6f} diversity={diversity_bonus:.6f} "
              f"novelty={novelty_bonus:.6f} improve={improvement_bonus:.6f} total={reward:.6f}")

        self.prev_action = action.copy()
        self.prev_metrics = new_metrics.copy()
        self.last_metrics = new_metrics

        next_observation = self._get_network_state_from_onos()

        self.step_count += 1
        terminated = self.step_count >= 200
        truncated = False

        info = {
            "throughput": throughput,
            "latency": latency,
            "packet_loss": packet_loss,
        }
        return next_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
