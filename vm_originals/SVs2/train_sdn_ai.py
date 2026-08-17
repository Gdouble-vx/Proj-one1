import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sdn_network_env import CustomSDNEnv
import os

if __name__ == "__main__":
    print("กำลังเริ่มสร้างสถาปัตยกรรมเครือข่ายจำลองและโหลด GNN...")
    env = CustomSDNEnv(vm1_ip="192.168.10.165", num_nodes=56)

    # 1. สร้าง Callback เพื่อเซฟโมเดลทุกๆ 10,000 steps (กันปัญหาหลุดกลางคัน)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ppo_sdn"
    )

    # กำหนดชื่อไฟล์หลักสำหรับเซฟและโหลดโมเดล
    MODEL_PATH = "ppo_gnn_sdn_model"
    total_timesteps = 100000 

    # 2. 🔥 [จุดที่แก้ไข] โลจิกตรวจสอบไฟล์เดิม: โฮลดเพื่อเทรนต่อ หรือสร้างใหม่
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"\n--- 🧠 พบไฟล์เดิม! กำลังโหลดน้ำหนักโมเดลจาก {MODEL_PATH}.zip เพื่อเทรนต่อ... ---")
        # โหลดโมเดลเก่ากลับมา และผูกเข้ากับ env ปัจจุบัน
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("\n--- 👶 ไม่พบไฟล์เดิม เริ่มต้นสร้างสถาปัตยกรรม PPO + GNN ใหม่จากศูนย์... ---")
        # ถ้าไม่มีไฟล์เก่า ให้ตั้งค่าโมเดลใหม่ตามปกติของคุณ
        model = PPO(
            "MlpPolicy", env,
            verbose=1,
            learning_rate=1e-4,
            batch_size=128,
            n_steps=1024,         # ต้องหารด้วย batch_size ลงตัว (1024/128 = 8)
            ent_coef=0.01,
            tensorboard_log="./ppo_sdn_tensorboard/",
            device="auto"         # ใช้ GPU ถ้ามี, ถ้าไม่มีใช้ CPU
        )

    print("\nเริ่มกระบวนการเทรนร่วมกับระบบ SDN...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            progress_bar=True, # แสดง Progress bar
            reset_num_timesteps=False # 🔥 [จุดที่เพิ่ม] ใส่ไว้เพื่อให้ตัวนับรอบ (Timesteps) เดินหน้าต่อจากเดิม ไม่รีเซ็ตเป็น 0
        )
        
        # เซฟโมเดลเมื่อเทรนสำเร็จตามเป้าหมาย
        model.save(MODEL_PATH)
        print(f"\nเทรนสำเร็จ! เซฟโมเดลเรียบร้อยในชื่อ {MODEL_PATH}.zip")
        
    except KeyboardInterrupt:
        # กด Ctrl+C ได้และโมเดลจะถูกเซฟทับตัวหลัก เพื่อให้ครั้งหน้าโหลดไปรันต่อได้เลย
        print("\nการเทรนถูกยกเลิกโดยผู้ใช้ (KeyboardInterrupt)")
        model.save(MODEL_PATH)
        print(f"เซฟสถานะล่าสุดเรียบร้อยในชื่อ {MODEL_PATH}.zip (คุณสามารถรันสคริปต์นี้ใหม่เพื่อเทรนต่อได้ทันที)")
    finally:
        env.close()
