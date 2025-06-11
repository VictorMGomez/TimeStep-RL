import csv
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from entorno_robot import EntornoRobot

def make_env():
    env = EntornoRobot()
    env.render_mode = True
    return env

if __name__ == "__main__":
    # 1) Creamos el env vectorizado (aunque solo sea uno, SB3 lo requiere así)
    env = DummyVecEnv([make_env])

    # 2) Cargamos el modelo
    print("Cargando modelo...")
    model = PPO.load("./ppo_runs/run_0/ppo_entorno_robot.zip", env=env)
    print("Modelo cargado correctamente")

    # 3) Reset inicial
    obs = env.reset()
    episode_num = 0
    results = []

    # 4) Bucle de pasos
    for step in range(100_000):
        print(f"Paso {step+1}: Prediciendo acción...")
        action, _states = model.predict(obs, deterministic=True)
        print(f"Acción tomada: {action}")

        obs, rewards, dones, infos = env.step(action)
        reward = rewards[0]
        done   = dones[0]
        info   = infos[0]

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
                "result":  result_type,
                "steps":   step + 1,
                "reward":  float(reward)
            })
            obs = env.reset()

    # 5) Guardamos resultados a CSV
    csv_file = "test_results.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "result", "steps", "reward"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Resultados guardados en {csv_file}")

    env.close()
    print("Prueba finalizada")
