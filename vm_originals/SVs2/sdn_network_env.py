import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import torch
import torch.nn as nn
import time
import subprocess
import re
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class NetworkGNNEncoder(nn.Module):
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


def calculate_reward(throughput, latency, packet_loss, alpha=1.2):
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


def measure_real_metrics(vm1_ip="192.168.10.165"):
    try:
        import requests as req
        response = req.get(f"http://{vm1_ip}:9999", timeout=5)
        data = response.json()
        return data["throughput"], data["latency"], data["packet_loss"]
    except Exception as e:
        print(f"[Metric] error: {e}")
        return 100.0, 50.0, 0.0

class CustomSDNEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, vm1_ip="192.168.10.165", num_nodes=14, gnn_output_dim=32):
        super(CustomSDNEnv, self).__init__()

        self.vm1_ip = vm1_ip
        self.num_nodes = num_nodes
        self.auth = HTTPBasicAuth('onos', 'rocks')
        self.max_links = 200
        self.step_count = 0
        self.prev_weights = {}
        self.use_real_metrics = True  # ← เปลี่ยนเป็น False ถ้า SSH ไม่ได้

        self.action_space = spaces.Box(low=1.0, high=100.0, shape=(self.max_links,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(gnn_output_dim,), dtype=np.float32)

        self.gnn_encoder = NetworkGNNEncoder(node_features=1, edge_features=2, output_dim=gnn_output_dim)
        self.gnn_encoder.eval()

    def _get_network_state_from_onos(self):
        devices_url = f"http://{self.vm1_ip}:8181/onos/v1/devices"
        links_url = f"http://{self.vm1_ip}:8181/onos/v1/links"

        try:
            devices_response = requests.get(devices_url, auth=self.auth, timeout=5)
            links_response = requests.get(links_url, auth=self.auth, timeout=5)

            if devices_response.status_code == 200 and links_response.status_code == 200:
                devices_data = devices_response.json()["devices"]
                links_data = links_response.json()["links"]
                device_map = {d["id"]: i for i, d in enumerate(devices_data)}

                src_list, dst_list = [], []
                edge_features = []

                for link in links_data:
                    src_id = link["src"]["device"]
                    dst_id = link["dst"]["device"]
                    if src_id in device_map and dst_id in device_map:
                        src_list.append(device_map[src_id])
                        dst_list.append(device_map[dst_id])
                        edge_features.append([0.1, 2.0])

                if len(src_list) == 0:
                    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
                    edge_attr = torch.tensor([[0.1, 2.0]] * 4, dtype=torch.float)
                else:
                    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                    edge_attr = torch.tensor(edge_features, dtype=torch.float)

                node_features = [[0.0] for _ in range(len(devices_data) if len(devices_data) > 0 else 3)]
                x = torch.tensor(node_features, dtype=torch.float)
            else:
                edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
                edge_attr = torch.tensor([[0.1, 2.0]] * 4, dtype=torch.float)
                x = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float)

            network_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            with torch.no_grad():
                state_vector = self.gnn_encoder(network_graph)

            return state_vector.numpy().flatten()

        except Exception as e:
            print(f"Error fetching data from ONOS: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_weights = {}
        observation = self._get_network_state_from_onos()
        return observation, {}

    def step(self, action):
        try:
            print(f"[AI Action - Step {self.step_count+1}/200] กำลังอัปเดต Link Weights ({len(action)} ลิงก์)...")
            links_url = f"http://{self.vm1_ip}:8181/onos/v1/links"
            links_response = requests.get(links_url, auth=self.auth, timeout=5)

            if links_response.status_code == 200:
                links_data = links_response.json().get("links", [])
                for idx, link in enumerate(links_data):
                    if idx >= len(action):
                       break
                    weight_value = float(action[idx])
                    prev = self.prev_weights.get(idx, None)

                     # ส่งเฉพาะที่เปลี่ยนมากกว่า 2.0 หรือครั้งแรก
                    if prev is None or abs(weight_value - prev) > 2.0:
                        src_device = link["src"]["device"]
                        src_port = link["src"]["port"]
                        dst_device = link["dst"]["device"]
                        dst_port = link["dst"]["port"]
                        config_url = f"http://{self.vm1_ip}:8181/onos/v1/network/configuration/links/{src_device}/{src_port}-{dst_device}/{dst_port}"
                        payload = {"annotations": {"cost": str(weight_value)}}
                        requests.post(config_url, json=payload, auth=self.auth, timeout=5)
                        self.prev_weights[idx] = weight_value

            print("[ONOS Status] อัปเดตค่าน้ำหนักเส้นทางหนีคอขวดเสร็จสิ้น")

        except Exception as e:
            print(f"Error sending action to ONOS: {e}")

        time.sleep(2.5)

        # วัดค่าจริงหรือ mock
        if self.use_real_metrics:
            throughput, latency, packet_loss = measure_real_metrics(
                vm1_ip=self.vm1_ip
            )
            print(f"[Metrics] Throughput={throughput:.1f}Mbps Latency={latency:.1f}ms Loss={packet_loss:.3f}")
        else:
            mean_weight = float(np.mean(action))
            throughput = float(np.clip(1000.0 - mean_weight * 2, 100, 1000))
            latency = float(np.clip(5.0 + mean_weight * 0.3, 1, 200))
            packet_loss = float(np.clip(mean_weight / 5000, 0, 0.1))

        reward = calculate_reward(throughput, latency, packet_loss)
        reward = reward / 1e5

        next_observation = self._get_network_state_from_onos()

        self.step_count += 1
        terminated = self.step_count >= 200
        truncated = False

        info = {
            "throughput": throughput,
            "latency": latency,
            "packet_loss": packet_loss
        }

        return next_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
