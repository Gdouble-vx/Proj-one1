"""
network_sim.py — Fast network simulator (pure NumPy, no deep-learning deps).

ใช้สำหรับจำลองเครือข่าย SDN 14 โหนด / 50 ลิงก์ เร็วมาก (ไม่ต้องเรียก REST API)
เพื่อให้ PPO เทรนได้หลายพัน step ต่อวินาที แล้วค่อย Fine-Tune กับ ONOS จริง
(ดู fine_tune_sdn_agent.py / custom_sdn_env.py)

แนวคิดการจำลอง:
  - Topology: กราฟเชื่อมต่อ 14 โหนด 50 ลิงก์ (random แต่ fix seed → reproducible)
  - Flows: 10 คู่ src-dst พร้อม demand (Mbps) สุ่มใหม่ทุก episode
  - Routing: Dijkstra ตาม link weight (ที่ agent เป็นคนกำหนด)
  - Bottleneck: ถ้า link ไหน utilization สูงเกิน threshold → มี packet loss + latency เพิ่ม
  - Metrics: aggregate throughput, average end-to-end latency, packet loss ratio

ECMP ใช้สำหรับ baseline แบบดั้งเดิม (split demand เท่า ๆ กันทุก shortest path)
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Reward (เหมือนกับใน CustomSDNEnv เดิมของ user)
# ----------------------------------------------------------------------------
def calculate_reward(throughput: float, latency: float, packet_loss: float,
                     alpha: float = 1.2, scale: float = 1e5) -> float:
    """Network Power + penalty เรื่อง packet loss (scale 1e5 ให้ค่า reward อยู่ในระดับ 10^-3)."""
    latency = max(float(latency), 0.001)
    network_power = (float(throughput) ** alpha) / latency
    penalty = 0.0
    if packet_loss > 0.05:
        penalty = -100.0
    elif packet_loss > 0.01:
        penalty = -20.0
    elif packet_loss > 0.001:
        penalty = -5.0
    return float((network_power + penalty) / scale)


# ----------------------------------------------------------------------------
# Simulator core (ไม่ขึ้นกับ gymnasium/torch → ทดสอบได้คนเดียว)
# ----------------------------------------------------------------------------
class NetworkSimulator:
    """จำลองโหลดของเครือข่ายตาม link weights ที่กำหนด"""

    UTIL_LOSS_THRESHOLD = 0.90   # เริ่มมี loss เมื่อ link ใช้เกิน 90%
    UTIL_DELAY_THRESHOLD = 0.85  # เริ่มมี queueing delay เมื่อใช้เกิน 85%

    def __init__(self, num_nodes: int = 14, num_links: int = 50, seed: int = 42,
                 link_capacity: float = 500.0, num_flows: int = 12,
                 demand_low: float = 100.0, demand_high: float = 400.0,
                 base_delay_ms: float = 2.0, max_episode_steps: int = 50):
        self.num_nodes = num_nodes
        self.num_links = num_links                # จำนวนลิงก์ (undirected)
        self.max_links = num_links                # ขนาด buffer ของ action/obs (ดู fine_tune_sdn_agent)
        self.link_capacity = float(link_capacity)
        self.num_flows = num_flows
        self.demand_low = float(demand_low)
        self.demand_high = float(demand_high)
        self.base_delay_ms = float(base_delay_ms)
        self.max_episode_steps = max_episode_steps
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.edges_u, self.edges_v = self._random_connected_graph(rng)  # (L,), (L,)
        self.edge_index_gnn = self._build_gnn_edge_index()              # (2, max_links) ใช้ใน GNN

        self.flows: List[Tuple[int, int, float]] = []   # list of (src, dst, demand)
        self.sample_flows(seed)                        # flows เริ่มต้น
        self.step_count = 0

    # ------------------------------------------------------------------ graph
    def _random_connected_graph(self, rng: np.random.Generator):
        """สร้าง connected graph จำนวน num_nodes โหนด num_links ลิงก์ (ไม่มี loop/duplicate)."""
        used: set = set()
        edges_u: List[int] = []
        edges_v: List[int] = []
        connected = [0]
        remaining = list(range(1, self.num_nodes))

        # Phase 1: spanning tree (รับประกันว่าเชื่อมกันหมด)
        while remaining:
            new_node = remaining.pop(rng.integers(len(remaining)))
            old_node = connected[rng.integers(len(connected))]
            edges_u.append(old_node)
            edges_v.append(new_node)
            used.add(frozenset((old_node, new_node)))
            connected.append(new_node)

        # Phase 2: เติมลิงก์เพิ่มจนครบ num_links
        tries = 0
        while len(edges_u) < self.num_links and tries < 10000:
            tries += 1
            u = int(rng.integers(self.num_nodes))
            v = int(rng.integers(self.num_nodes))
            if u == v:
                continue
            key = frozenset((u, v))
            if key in used:
                continue
            edges_u.append(u)
            edges_v.append(v)
            used.add(key)

        if len(edges_u) < self.num_links:
            raise ValueError(
                f"ไม่สามารถสร้างกราฟ {self.num_nodes} โหนด {self.num_links} ลิงก์ได้ "
                f"(สร้างได้ {len(edges_u)} — ลิงก์มากเกินไป)")

        return np.array(edges_u, dtype=np.int64), np.array(edges_v, dtype=np.int64)

    def _build_gnn_edge_index(self) -> np.ndarray:
        """GNN ใช้ directed edge (src -> dst) ครั้งละลิงก์; pad ด้วย self-loop (0,0)
        ให้ได้ขนาด (2, max_links) ตรงกับ action space เสมอ."""
        n = len(self.edges_u)
        idx = np.zeros((2, self.max_links), dtype=np.int64)
        for i in range(self.max_links):
            if i < n:
                idx[0, i] = int(self.edges_u[i])
                idx[1, i] = int(self.edges_v[i])
            else:
                idx[0, i] = 0
                idx[1, i] = 0
        return idx

    # ------------------------------------------------------------------ flows
    def sample_flows(self, seed: Optional[int] = None, num_flows: Optional[int] = None) -> None:
        """สุ่ม flows ใหม่ (src, dst, demand Mbps) — seed เดียวกันได้ scenario เดียวกัน."""
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        k = num_flows if num_flows is not None else self.num_flows
        flows = []
        for _ in range(k):
            src = int(rng.integers(self.num_nodes))
            dst = int(rng.integers(self.num_nodes))
            while dst == src:
                dst = int(rng.integers(self.num_nodes))
            demand = float(rng.uniform(self.demand_low, self.demand_high))
            flows.append((src, dst, demand))
        self.flows = flows

    # ----------------------------------------------------------------- routing
    def dijkstra_path(self, src: int, dst: int, weights: Sequence[float]) -> List[int]:
        """หาเส้นทางที่สั้นที่สุด (ตาม weights) ระหว่าง src-dst → list ของ link index."""
        if src == dst:
            return []
        n = self.num_nodes
        adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(n)]
        for i, (u, v) in enumerate(zip(self.edges_u, self.edges_v)):
            w = float(weights[i])
            adj[u].append((v, i, w))
            adj[v].append((u, i, w))

        dist = [float("inf")] * n
        prev: List[Optional[int]] = [None] * n
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for v, li, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = li
                    heapq.heappush(heap, (nd, v))

        if dist[dst] == float("inf"):
            return []
        # ย้อนรอยหา path
        path: List[int] = []
        cur = dst
        while cur != src:
            li = prev[cur]
            if li is None:
                return []
            path.append(li)
            # หาโหนดก่อนหน้า
            u, v = int(self.edges_u[li]), int(self.edges_v[li])
            cur = v if cur == u else u
        path.reverse()
        return path

    def _all_shortest_hop_paths(self, src: int, dst: int) -> List[List[int]]:
        """หาเส้นทางทั้งหมดที่มี hop น้อยที่สุด (ใช้กับ ECMP)."""
        n = self.num_nodes
        adj: List[List[int]] = [[] for _ in range(n)]
        for i, (u, v) in enumerate(zip(self.edges_u, self.edges_v)):
            adj[u].append((v, i))
            adj[v].append((u, i))

        dist = [-1] * n
        dist[src] = 0
        prev: List[List[Tuple[int, int]]] = [[] for _ in range(n)]  # (node, link)
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v, li in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v].append((u, li))
                    queue.append(v)
                elif dist[v] == dist[u] + 1:
                    prev[v].append((u, li))

        # DFS ย้อนรอย
        def dfs(node: int) -> List[List[int]]:
            if node == src:
                return [[]]
            out = []
            for pnode, pli in prev[node]:
                for sub in dfs(pnode):
                    out.append(sub + [pli])
            return out

        if dist[dst] == -1:
            return [[]]
        return dfs(dst)

    # ----------------------------------------------------------------- metrics
    def simulate(self, weights: Sequence[float]) -> dict:
        """รัน routing ตาม weights แล้วคืน metrics (dict) ของรอบนี้."""
        paths = [self.dijkstra_path(f[0], f[1], weights) for f in self.flows]
        return self._compute_metrics(paths)

    def simulate_ecmp(self) -> dict:
        """Baseline ECMP: split demand เท่า ๆ กันทุก shortest-hop path."""
        all_paths = [self._all_shortest_hop_paths(f[0], f[1]) for f in self.flows]
        paths = []
        for f, cands in zip(self.flows, all_paths):
            good = [p for p in cands if p]
            paths.append(good if good else [[]])
        return self._compute_metrics(paths, split_ecmp=True)

    def _compute_metrics(self, paths: List[List[int]], split_ecmp: bool = False) -> dict:
        # 1) โหลดแต่ละลิงก์
        link_load = np.zeros(self.num_links, dtype=np.float64)
        if split_ecmp:
            for f, cands in zip(self.flows, paths):
                split = f[2] / max(len(cands), 1)
                for p in cands:
                    for li in p:
                        link_load[li] += split
        else:
            for f, p in zip(self.flows, paths):
                for li in p:
                    link_load[li] += f[2]

        utilization = link_load / self.link_capacity
        # loss ต่อลิงก์: 0 ถ้าใช้ < 90% แล้วเพิ่มขึ้นเชิงเส้นจนถึง 30%
        link_loss = np.clip((utilization - self.UTIL_LOSS_THRESHOLD) * 1.2, 0.0, 0.30)

        total_demand = float(sum(f[2] for f in self.flows))
        delivered = 0.0
        latencies: List[float] = []

        for f, cands in zip(self.flows, paths):
            if split_ecmp:
                n_paths = max(len(cands), 1)
                ratio = 1.0 / n_paths
                for sub in cands:
                    if not sub:
                        delivered += f[2] * ratio          # เส้นทางว่าง = ส่งถึงหมด
                        latencies.append(0.0)
                        continue
                    loss = 1.0 - float(np.prod(1.0 - link_loss[sub]))
                    delivered += f[2] * ratio * (1.0 - loss)
                    latencies.append(ratio * self._path_latency(sub, utilization))
            else:
                p = cands or []
                if p:
                    loss = 1.0 - float(np.prod(1.0 - link_loss[p]))
                    delivered += f[2] * (1.0 - loss)
                else:
                    delivered += f[2]                      # src == dst (เส้นทางว่าง)
                latencies.append(self._path_latency(p, utilization) if p else 0.0)

        throughput = float(delivered)
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        raw_loss = 1.0 - (throughput / total_demand) if total_demand > 0 else 0.0
        packet_loss = float(np.clip(raw_loss, 0.0, 1.0))

        return {
            "throughput": throughput,
            "latency": avg_latency,
            "packet_loss": packet_loss,
            "link_utilization": utilization,
            "link_load": link_load,
            "reward": calculate_reward(throughput, avg_latency, packet_loss),
        }

    def _path_latency(self, path: List[int], utilization: np.ndarray) -> float:
        lat = 0.0
        for li in path:
            u = float(utilization[li])
            lat += self.base_delay_ms
            if u > self.UTIL_DELAY_THRESHOLD:
                lat += (u - self.UTIL_DELAY_THRESHOLD) * 30.0  # queueing delay
        return lat

    # ------------------------------------------------------------- observation
    def build_observation(self, weights: np.ndarray, utilization: np.ndarray) -> np.ndarray:
        """obs = node_feat(num_nodes) + edge_attr(max_links*2)
        node_feat  = max utilization ของลิงก์ที่ชนกับโหนดนั้น
        edge_attr  = [utilization, normalized_weight] ต่อ directed edge (pad self-loop = 0)"""
        node_feat = np.zeros(self.num_nodes, dtype=np.float32)
        for i, (u, v) in enumerate(zip(self.edges_u, self.edges_v)):
            node_feat[u] = max(node_feat[u], float(utilization[i]))
            node_feat[v] = max(node_feat[v], float(utilization[i]))

        n = len(self.edges_u)
        edge_attr = np.zeros((self.max_links, 2), dtype=np.float32)
        for i in range(self.max_links):
            if i < n:
                w = float(weights[i])
                edge_attr[i, 0] = float(utilization[i])
                edge_attr[i, 1] = min(w / 100.0, 1.0)
            else:
                edge_attr[i, 0] = 0.0
                edge_attr[i, 1] = 0.0
        return np.concatenate([node_feat, edge_attr.reshape(-1)]).astype(np.float32)

    # ------------------------------------------------------------ diagnostics
    def describe_topology(self) -> str:
        deg = np.zeros(self.num_nodes, dtype=int)
        for u, v in zip(self.edges_u, self.edges_v):
            deg[u] += 1
            deg[v] += 1
        return (f"Topology: {self.num_nodes} nodes, {self.num_links} links "
                f"(degree min={deg.min()}, max={deg.max()}, avg={deg.mean():.1f})")
