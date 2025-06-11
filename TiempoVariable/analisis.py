import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean_monitor_csv(fn):
    cleaned = []
    dts = []
    with open(fn, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue  # Saltar línea bug
            try:
                r = float(parts[0])
                l = int(parts[1])
                t = float(parts[2])
                # Opcional: leer mean_dt si existe
                mean_dt = float(parts[3]) if len(parts) > 3 else None
                # Filtro de reward (ajusta margen si lo necesitas)
                if not any(abs(r - val) < 0.05 for val in (100, 0, -100)):
                    continue
                cleaned.append([r, l, t])
                if mean_dt is not None:
                    dts.append(mean_dt)
            except Exception:
                continue
    df = pd.DataFrame(cleaned, columns=["r", "l", "t"])
    return df, dts


def collect_results(log_dir):
    all_rewards = []
    all_dts     = []
    all_steps   = []
    run_dirs = sorted([
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("run_")
    ])
    for run in run_dirs:
        fn = os.path.join(log_dir, run, "monitor.csv")
        df, dts = clean_monitor_csv(fn)
        timesteps = df['l'].cumsum().values
        rewards   = df['r'].values
        arr = np.vstack([timesteps, rewards]).T
        if arr.shape[0] > 0:
            all_rewards.append(arr)
        if dts:
            all_dts.append(np.array(dts))
        if 'l' in df.columns and len(df['l']) > 0:
            all_steps.append(df['l'].values)
    return all_rewards, all_dts, all_steps


def average_curve(all_rewards, step=10_000):
    """Interpola recompensa media cada 'step' timesteps, devuelve (ts, μ, σ)."""
    all_rewards = [arr for arr in all_rewards if arr.shape[0] > 0]  # <-- FILTRO EXTRA
    if len(all_rewards) == 0:
        raise ValueError("No se han encontrado runs con datos válidos (tras filtrar arrays vacíos).")

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
    LOGDIR = "./ppo_runs"
    all_r, all_dts, all_steps = collect_results(LOGDIR)

    # FILTRAR arrays vacíos (¡aquí el fix principal!)
    all_r = [arr for arr in all_r if arr.shape[0] > 0]
    if len(all_r) == 0:
        raise ValueError("No hay runs con datos válidos (todos los arrays vacíos).")
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
        plt.xlabel("Episodio (acumulado en todas las ejecuciones)")
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
