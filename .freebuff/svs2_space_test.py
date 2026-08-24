import numpy as np
from custom_sdn_env import CustomSDNEnv
from stable_baselines3 import PPO

env = CustomSDNEnv(vm1_ip="192.168.10.165", num_nodes=14, max_links=21,
                   obs_mode="gnn", use_real_metrics=True, step_delay=0)
print("env obs_space:", env.observation_space)
print("env action_space:", env.action_space)

model = PPO.load("/home/ino/sdn-ai-brain/ppo_gnn_sdn_model.zip", env=env, device="auto")
print("model obs_space:", model.observation_space)
print("model action_space:", model.action_space)

obs, _ = env.reset()
print("obs shape:", obs.shape, "dtype:", obs.dtype, "finite:", np.isfinite(obs).all())
action, _ = model.predict(obs, deterministic=True)
print("action shape:", action.shape, "range:", float(action.min()), "-", float(action.max()))
obs2, r, term, trunc, info = env.step(action)
print("STEP OK: reward=", r, "info=", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
match = (env.observation_space.shape == model.observation_space.shape
         and env.action_space.shape == model.action_space.shape)
print("VERDICT: spaces MATCH" if match else "VERDICT: MISMATCH")
env.close()
