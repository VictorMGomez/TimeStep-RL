import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob

def load_all_monitor_csvs(base_dir="ppo_runs_2s"):
    """
    Carga todos los archivos monitor.csv en subcarpetas run_* de base_dir.
    Devuelve un único DataFrame con columna 'run_id' indicando la run original.
    """
    all_dfs = []
    run_paths = sorted(Path(base_dir).glob("run_*/monitor.csv"))
    for i, file_path in enumerate(run_paths):
        # Leer metadatos
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith("#"):
                try:
                    metadata = json.loads(first_line[1:])
                except Exception:
                    metadata = {}
            else:
                metadata = {}
        # Leer CSV
        try:
            df = pd.read_csv(file_path, skiprows=1)
        except Exception as e:
            print(f"[WARN] No se pudo leer {file_path}: {e}")
            continue
        # Marcar run id y metadatos
        df["run_id"] = i
        df["run_path"] = str(file_path.parent)
        all_dfs.append(df)
    if not all_dfs:
        print(f"No se encontró ningún monitor.csv válido en {base_dir}")
        return None
    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all

def analyze_multiple_runs(base_dir="ppo_runs_2s"):
    df = load_all_monitor_csvs(base_dir)
    if df is None:
        print("No hay datos para analizar.")
        return None

    # Estadísticas globales por episodio
    print("=== ESTADÍSTICAS AGREGADAS DE TODAS LAS RUNS ===\n")
    steps_stats = df['l'].describe()
    print(f"Total de episodios analizados: {len(df)}")
    print(f"Promedio de pasos por episodio: {steps_stats['mean']:.2f}")
    print(f"Mediana de pasos: {steps_stats['50%']:.2f}")
    print(f"Desviación estándar: {steps_stats['std']:.2f}")
    print(f"Min: {int(steps_stats['min'])}, Max: {int(steps_stats['max'])}")
    print(f"Rango intercuartílico (Q3-Q1): {steps_stats['75%'] - steps_stats['25%']:.2f}\n")

    reward_stats = df['r'].describe()
    print(f"Recompensa promedio: {reward_stats['mean']:.2f}")
    print(f"Mediana de recompensa: {reward_stats['50%']:.2f}")
    print(f"Máxima recompensa: {reward_stats['max']:.2f}")
    print(f"Mínima recompensa: {reward_stats['min']:.2f}\n")

    # Analizar tendencia promedio entre runs (rolling mean promedio por episodio)
    df['episode_in_run'] = df.groupby('run_id').cumcount() + 1
    # Para cada episodio (en el eje X), calcular la media y desviación entre runs
    pivot = df.pivot_table(index='episode_in_run', values='l', aggfunc=['mean', 'std', 'count'])
    rolling = pivot['mean'].rolling(window=10, min_periods=1).mean()

    # Graficar
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis Agregado de Entrenamiento PPO', fontsize=16, fontweight='bold')

    # 1. Evolución promedio de pasos por episodio (entre runs)
    axes[0, 0].plot(pivot.index, pivot['mean'], label='Promedio pasos por episodio', color='b')
    axes[0, 0].fill_between(pivot.index, pivot['mean'] - pivot['std'], pivot['mean'] + pivot['std'], 
                            alpha=0.3, label='±1σ')
    axes[0, 0].plot(pivot.index, rolling, 'r-', linewidth=2, label='Promedio móvil (10)')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Pasos')
    axes[0, 0].set_title('Evolución de pasos por episodio (media entre runs)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Histograma de pasos por episodio
    axes[0, 1].hist(df['l'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(df['l'].mean(), color='red', linestyle='--', linewidth=2, label=f'Promedio: {df["l"].mean():.1f}')
    axes[0, 1].set_xlabel('Pasos por Episodio')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución global de pasos')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Evolución de recompensas (media entre runs)
    reward_pivot = df.pivot_table(index='episode_in_run', values='r', aggfunc=['mean', 'std'])
    reward_rolling = reward_pivot['mean'].rolling(window=10, min_periods=1).mean()
    axes[1, 0].plot(reward_pivot.index, reward_pivot['mean'], label='Promedio recompensa', color='g')
    axes[1, 0].fill_between(reward_pivot.index, reward_pivot['mean'] - reward_pivot['std'], 
                            reward_pivot['mean'] + reward_pivot['std'], alpha=0.3, label='±1σ')
    axes[1, 0].plot(reward_pivot.index, reward_rolling, 'orange', linewidth=2, label='Promedio móvil (10)')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Recompensa')
    axes[1, 0].set_title('Evolución de recompensa (media entre runs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Relación pasos vs recompensa global
    axes[1, 1].scatter(df['l'], df['r'], alpha=0.4, color='purple')
    axes[1, 1].set_xlabel('Pasos por Episodio')
    axes[1, 1].set_ylabel('Recompensa')
    axes[1, 1].set_title('Relación Pasos vs Recompensa')
    correlation = df['l'].corr(df['r'])
    axes[1, 1].text(0.05, 0.95, f'Correlación: {correlation:.3f}', 
                    transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n=== RESUMEN FINAL (Todas las runs) ===")
    print(f"• Total episodios: {len(df)} en {df['run_id'].nunique()} runs")
    print(f"• Pasos promedio por episodio: {df['l'].mean():.1f}")
    print(f"• Rango: {int(df['l'].min())}-{int(df['l'].max())} pasos")
    print(f"• Recompensa promedio: {df['r'].mean():.2f}")

    return df

# Ejecutar análisis agregado
if __name__ == "__main__":
    base_dir = "ppo_runs_2s"   # Cambia aquí la carpeta base de tus runs
    print(f"Analizando todas las runs en {base_dir}")
    data = analyze_multiple_runs(base_dir)
    if data is not None:
        print("\n¡Análisis global completado!")
    else:
        print("\nNo se pudo completar el análisis.")
