import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_monitor_csv(fn):
    """Lee el monitor.csv y devuelve un DataFrame solo con episodios válidos:
       - Quita líneas con columnas incorrectas
       - Solo acepta r ∈ {100, 0, -100}
    """
    cleaned = []
    with open(fn, "r") as f:
        for line in f:
            # Saltar cabeceras
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue  # Saltar línea bug
            try:
                r = float(parts[0])
                l = int(parts[1])
                t = float(parts[2])
                if r not in (100, 0, -100):
                    continue  # Saltar episodios con reward raro
                cleaned.append([r, l, t])
            except Exception:
                continue  # Si hay algún error raro, ignorar línea
    # DataFrame limpio
    df = pd.DataFrame(cleaned, columns=["r", "l", "t"])
    return df

def collect_results(log_dir):
    all_rewards = []
    all_dts     = []  # añadimos lista para mean_dt por episodio
    all_steps   = []  # pasos por episodio
    run_dirs = sorted([
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("run_")
    ])
    for run in run_dirs:
        fn = os.path.join(log_dir, run, "monitor.csv")
        df = clean_monitor_csv(fn)
        # construimos un array (timesteps_acumulados, reward)
        timesteps = df['l'].cumsum().values
        rewards   = df['r'].values
        all_rewards.append(np.vstack([timesteps, rewards]).T)
        # Para dt y pasos por episodio, si tienes mean_dt en otras columnas, puedes añadirlo aquí
        # if 'mean_dt' in df.columns: ...
        if 'l' in df.columns:
            all_steps.append(df['l'].values)
    return all_rewards, all_dts, all_steps

def average_curve(all_rewards, step=10_000):
    """Interpola recompensa media cada 'step' timesteps, devuelve (ts, μ, σ)."""
    if len(all_rewards) == 0:
        raise ValueError("No se han encontrado runs con datos válidos.")

    max_t = max(np.nanmax(arr[:, 0]) for arr in all_rewards)
    if not np.isfinite(max_t):
        raise ValueError(f"max_t no es finito: {max_t!r}")
    max_t = int(max_t)

    grid = np.arange(0, max_t + step, step, dtype=int)

    interp_rs = []
    for arr in all_rewards:
        t = arr[:, 0]
        r = arr[:, 1]
        valid = ~np.isnan(t)
        t = t[valid]
        r = r[valid]
        bins = np.digitize(t, grid)
        mean_r = [
            r[bins == k].mean() if np.any(bins == k) else np.nan
            for k in range(1, len(grid))
        ]
        interp_rs.append(mean_r)
    R = np.vstack(interp_rs)     # shape: (n_runs, n_bins)
    mu    = np.nanmean(R, axis=0)
    sigma = np.nanstd (R, axis=0)
    return grid[1:], mu, sigma

if __name__ == "__main__":
    LOGDIR = "./ppo_runs_0.5s"
    all_r, all_dts, all_steps = collect_results(LOGDIR)
    ts, mu, sigma = average_curve(all_r, step=15_000)

    final_mean_reward = mu[-1]          # reward medio en el último punto de la curva
    print(f"Reward final medio: {final_mean_reward:.2f}")

    # 1. Reward medio vs timesteps
    plt.figure(figsize=(8,5))
    plt.plot(ts, mu, label="Reward medio")
    plt.fill_between(ts, mu - sigma, mu + sigma, alpha=0.3, label="±1σ")
    plt.xlabel("Timesteps totales")
    plt.ylabel("Reward por episodio")
    plt.title("Curva media de reward vs. timesteps")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. dt medio por episodio (si tienes la columna)
    if all_dts and all_steps:
        # Concatenamos todas las runs para una gráfica general
        all_mean_dt = np.concatenate(all_dts)
        all_steps_flat = np.concatenate(all_steps)
        plt.figure()
        plt.plot(all_mean_dt, label="dt medio por episodio")
        plt.xlabel("Episodio (acumulado en todas las runs)")
        plt.ylabel("dt medio")
        plt.title("Evolución de dt medio por episodio")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 3. Relación dt medio vs longitud episodio
        plt.figure()
        plt.scatter(all_mean_dt, all_steps_flat, alpha=0.4)
        plt.xlabel("dt medio del episodio")
        plt.ylabel("Pasos (steps) en episodio")
        plt.title("Relación entre dt medio y pasos por episodio")
        plt.tight_layout()
        plt.show()

        # 4. Estadísticas resumen
        print(f"dt medio (promedio de todos los episodios): {all_mean_dt.mean():.3f}")
        print(f"dt medio (desviación estándar): {all_mean_dt.std():.3f}")
        print(f"Pasos medios por episodio: {all_steps_flat.mean():.1f}")
        print(f"Pasos std por episodio: {all_steps_flat.std():.1f}")

    else:
        print("No se encontró columna 'mean_dt' o 'l' en los CSVs para análisis de dt.")

