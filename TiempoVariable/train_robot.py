import os
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from entorno_robot import EntornoRobot
from action_logger import ActionLoggerCallback

# Número de ejecuciones independientes
NUM_RUNS = 4
# Directorio base para todas las corridas
BASE_LOG_DIR = "./ppo_runs"

# Constantes de tu currículum y entrenamiento
TOTAL_TIMESTEPS = 400_000
MIN_OBS, MAX_OBS = 0, 10

class BlockStatsCallback(BaseCallback):
    """
    Cada block_size timesteps computa la media y σ de reward por episodio
    y lo graba en Tensorboard (o en un .csv si prefieres).
    """
    def __init__(self, block_size: int = 10_000, verbose=0):
        super().__init__(verbose)
        self.block_size = block_size
        self.next_threshold = block_size
        self.block_episode_rewards = []  # rewards acumulados de cada episodio del bloque
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 nos va dando reward por paso en self.locals["rewards"]
        r = float(self.locals["rewards"][0])
        self._current_ep_reward += r

        # Si el episodio terminó, lo guardamos
        if self.locals["dones"][0]:
            self.block_episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

        # Si superamos el siguiente umbral de timesteps...
        if self.num_timesteps >= self.next_threshold:
            if len(self.block_episode_rewards) > 0:
                mu    = np.mean(self.block_episode_rewards)
                sigma = np.std(self.block_episode_rewards)
                # grabar en tensorboard
                self.logger.record("block/reward_mean",    mu,   exclude=("stdout",))
                self.logger.record("block/reward_std",     sigma,exclude=("stdout",))
            # reset bloque
            self.block_episode_rewards.clear()
            self.next_threshold += self.block_size

        return True

class CurriculumCallback(BaseCallback):
    def __init__(self):
        super().__init__()
    def _on_step(self) -> bool:
        n_tramos = 5
        obs_por_tramo = 2
        frac = self.num_timesteps / TOTAL_TIMESTEPS
        tramo_actual = int(np.floor(frac * n_tramos))
        current = int(np.clip(MIN_OBS + obs_por_tramo * tramo_actual, MIN_OBS, MAX_OBS))
        # Desenrollar wrappers hasta el entorno base
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        env.num_obstacles = current
        self.logger.record("curriculum/num_obstacles", current)
        # Imprime cada 10k steps
        if self.num_timesteps % 10000 == 0:
            print(f"[Curriculum] Timesteps: {self.num_timesteps} - Obstáculos: {current}")
        return True

if __name__ == '__main__':
    visualizar = False

    for run_id in range(NUM_RUNS):
        # 1) Preparar directorio de esta corrida
        run_dir = os.path.join(BASE_LOG_DIR, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        seed = 1000 + run_id

        # 2) Constructor de entornos, con semilla y Monitor apuntando a monitor.csv
        def make_env():
            env = EntornoRobot(MIN_OBS)
            env.reset(seed=seed)
            if visualizar:
                env.render_mode = True
            return Monitor(env, filename=os.path.join(run_dir, "monitor.csv"), info_keywords=("mean_dt","std_dt"))

        # 3) Crear y normalizar train_env
        train_env = DummyVecEnv([make_env])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

        # 4) Crear eval_env (no entrena stats) y compartir obs_rms
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = train_env.obs_rms

        # 5) Callbacks: evaluación y logging de acciones
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model/"),
            log_path=os.path.join(run_dir, "eval/"),
            eval_freq=10_000,
            deterministic=True,
            render=False
        )
        action_logger = ActionLoggerCallback(verbose=1)
        curriculum_cb = CurriculumCallback()

        # 6) “Warm-up” de VecNormalize: 500 pasos aleatorios
        obs = train_env.reset()
        for _ in range(500):
            action = train_env.action_space.sample()
            obs, rewards, dones, infos = train_env.step([action])
            if dones[0]:
                obs = train_env.reset()

        # 7) Hiper-parámetros PPO
        policy_kwargs = dict(log_std_init=-1.0)
        ppo_kwargs = dict(
            learning_rate=2e-4,
            n_steps=2048,
            ent_coef=0.01,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            tensorboard_log=os.path.join(run_dir, "tensorboard/"),
            verbose=1,
            policy_kwargs=policy_kwargs
        )

        # 8) Cargar o crear modelo
        model_path = os.path.join(run_dir, "ppo_entorno_robot.zip")
        if os.path.exists(model_path):
            print(f"[Run {run_id}] Cargando modelo previo…")
            model = PPO.load(model_path, env=train_env, **ppo_kwargs)
        else:
            print(f"[Run {run_id}] Creando nuevo modelo…")
            model = PPO("MlpPolicy", train_env, **ppo_kwargs)

        # 9) Entrenar
        block_cb = BlockStatsCallback(block_size=10_000)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[action_logger, curriculum_cb, block_cb, eval_callback],
            reset_num_timesteps=False
        )

        # 10) Guardar normalizador y modelo final
        train_env.save(os.path.join(run_dir, "vecnorm.pkl"))
        model.save(model_path)

        train_env.close()
        eval_env.close()
        print(f"=== Run {run_id} completada ===\n")
