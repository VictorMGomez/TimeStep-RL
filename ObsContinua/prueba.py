import gym
import torch
from stable_baselines3 import PPO
from entorno_robot import EntornoRobot
from robot_diferencial import RobotDiferencial
import numpy as np


print("Cargando modelo...")
model = PPO.load("ppo_entorno_robot")
print("Modelo cargado correctamente")

# Crear entorno
env = EntornoRobot()
env.render_mode = True 
obs, _ = env.reset()
print("Entorno inicializado correctamente")

# Ejecutar prueba
for step in range(10000):  
    print(f"Paso {step+1}: Prediciendo acción...")
    action, _states = model.predict(obs, deterministic=True)
    print(f"Acción tomada: {action}")

    obs, reward, done, _, _ = env.step(action)
    print(f"Recompensa: {reward}, Done: {done}")

    env.render()  # Activamos renderizado

    if done:
        print("Episodio terminado, reiniciando entorno...")
        obs, _ = env.reset()

env.close()
print("Prueba finalizada")


