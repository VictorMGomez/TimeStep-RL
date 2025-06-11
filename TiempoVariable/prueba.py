import gym
import csv
import time
import torch
from stable_baselines3 import PPO
from entorno_robot import EntornoRobot
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np


def make_env():
    env = EntornoRobot()
    env.render_mode = True
    return env

raw_env = DummyVecEnv([make_env])
env = VecNormalize.load("vecnorm.pkl", raw_env)
env.training = False
env.norm_reward = False

print("Cargando modelo...")
model = PPO.load("./ppo_entorno_robot.zip", env=env)
print("Modelo cargado correctamente")

# Crear entorno
"""
env = EntornoRobot()
env.render_mode = True 
obs, _ = env.reset()
print("Entorno inicializado correctamente")
"""

# Ejecutar prueba
obs = env.reset()
episode_num = 0
results = []  # Lista para almacenar resultados episodios
print("Entorno inicializado correctamente")
for step in range(100000):  
    print(f"Paso {step+1}: Prediciendo acción...")
    action, _states = model.predict(obs, deterministic=True)
    print(f"Acción tomada: {action}")

    obs, rewards, dones, infos = env.step(action)
    reward = rewards[0]
    done = dones[0]
    info = infos[0]
    truncated = infos[0].get("truncated", False)
    #print(f"Recompensa: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    if done:
        episode_num += 1
        if info.get("truncated", False):
            result_type = "timeout"
        elif reward > 0:
            result_type = "llegada"
        else:
            result_type = "choque"
        
        results.append({
            "episode": episode_num,
            "result": result_type,
            "steps": step + 1,
            "reward": float(reward)
        })
        obs = env.reset()

csv_file = "test_results.csv"
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["episode", "result", "steps", "reward"])
    writer.writeheader()
    writer.writerows(results)
print(f"Resultados guardados en {csv_file}")

env.close()
print("Prueba finalizada")


