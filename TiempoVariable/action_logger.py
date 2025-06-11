import os
import csv
from stable_baselines3.common.callbacks import BaseCallback

class ActionLoggerCallback(BaseCallback):
    def __init__(self,
                 log_dir="logs",
                 csv_filename="terminaciones.csv",
                 steps_csv_filename="pasos.csv",
                 verbose=0):
        super().__init__(verbose)
        # Acumuladores por episodio
        self.episodio_rewards = []
        self.episodio_dt      = []
        self.episodio_time    = []
        self.episodio_v       = []
        self.episodio_length  = 0

        # Directorio y rutas
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # CSV de resultados episodios
        self.csv_path = os.path.join(self.log_dir, csv_filename)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "resultado"])

        # CSV de pasos
        self.steps_csv_path = os.path.join(self.log_dir, steps_csv_filename)
        if not os.path.exists(self.steps_csv_path):
            with open(self.steps_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "tiempo_transcurrido", "dt_elegido", "velocidad"])

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        dones   = self.locals.get("dones")

        if infos is None:
            return True

        info0     = infos[0]
        reward0   = rewards[0]
        dt_chosen = info0.get("dt_elegido")
        time_real = info0.get("tiempo_transcurrido")
        speed     = info0.get("velocidad")
        step      = self.num_timesteps

        # ——— Primero: grabamos el PASO en pasos.csv ———
        with open(self.steps_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, time_real, dt_chosen, speed])

        # ——— Ahora, acumulamos para el episodio ———
        self.episodio_length += 1
        self.episodio_rewards.append(reward0)
        if dt_chosen is not None:
            self.episodio_dt.append(dt_chosen)
            self.logger.record("custom/dt_elegido", dt_chosen)
        if time_real is not None:
            self.episodio_time.append(time_real)
            self.logger.record("custom/tiempo_real", time_real)

        # ——— Si terminó el episodio, volcamos en terminaciones.csv ———
        if dones[0]:
            # Determinar resultado
            if info0.get("truncated", False):
                resultado = "timeout"
            elif sum(self.episodio_rewards) >= 100:
                resultado = "llegada"
            elif sum(self.episodio_rewards) <= -100:
                resultado = "choque"
            else:
                resultado = "otro"

            # Escribir en terminaciones.csv
            episode = step  # o tu propio contador de episodios
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode, resultado])

            # Registrar en TensorBoard también
            total_time = sum(self.episodio_time) if self.episodio_time else 0.0
            avg_dt     = (sum(self.episodio_dt) / len(self.episodio_dt)) if self.episodio_dt else 0.0

            self.logger.record("episodio/recompensa_total", sum(self.episodio_rewards))
            self.logger.record("episodio/duracion", self.episodio_length)
            self.logger.record("episodio/media_dt", avg_dt)
            self.logger.record("episodio/total_time_real", total_time)
            self.logger.record("episodio/resultado_llegada", 1 if resultado == "llegada" else 0)
            self.logger.record("episodio/resultado_choque", 1 if resultado == "choque" else 0)
            self.logger.record("episodio/resultado_timeout", 1 if resultado == "timeout" else 0)

            # Reiniciar acumuladores de episodio
            self.episodio_rewards = []
            self.episodio_dt      = []
            self.episodio_time    = []
            self.episodio_length  = 0

        return True
