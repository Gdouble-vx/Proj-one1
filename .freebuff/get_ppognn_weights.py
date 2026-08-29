#!/usr/bin/env python3
"""Get PPO+GNN weights from SVs2. Run on SVs2."""
import sys, json, numpy as np
sys.path.insert(0, '/home/ino')
from stable_baselines3 import PPO
from train_gnn_v2 import ShapedFastSDNEnv, SDNGraphFeatureExtractor

env = ShapedFastSDNEnv(topology='nsfnet', seed=42)
model = PPO.load('/home/ino/ppo_gnn_50k_best')
obs, _ = env.reset(seed=0)
action, _ = model.predict(obs, deterministic=True)
w = np.clip(action[:21].astype(float), 1.0, 100.0)
edges = [(1,2),(1,3),(1,8),(2,3),(2,7),(3,4),(4,5),(4,6),(5,6),(5,7),
         (6,13),(6,14),(7,8),(8,9),(9,10),(9,12),(10,11),(10,13),(11,12),(11,14),(12,13)]
rules = []
for i,(u,v) in enumerate(edges):
    rules.append({'src': f'of:00000000000000{u:02x}', 'dst': f'of:00000000000000{v:02x}', 'weight': round(float(w[i]),2)})
print(json.dumps(rules))
