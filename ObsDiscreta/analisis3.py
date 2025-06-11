import pandas as pd
import matplotlib.pyplot as plt

def analizar_terminaciones(file_path="terminaciones.csv"):
    # Cargar el archivo CSV
    df = pd.read_csv(file_path)
    
    # Estadísticas básicas
    total = len(df)
    counts = df['resultado'].value_counts()
    porcentajes = counts / total * 100
    
    print("=== Estadísticas de terminaciones de episodios ===\n")
    for res, count in counts.items():
        print(f"- {res.capitalize()}: {count} episodios ({porcentajes[res]:.1f}%)")
    print(f"\nTotal de episodios analizados: {total}")
    
    # Mostrar evolución temporal
    plt.figure(figsize=(10,5))
    plt.plot(df['episode'], df['resultado'].map({'choque':0, 'timeout':1, 'llegada':2}), 'o', alpha=0.5)
    plt.yticks([0,1,2], ['Choque', 'Timeout', 'Llegada'])
    plt.xlabel('Número de Episodio')
    plt.ylabel('Resultado')
    plt.title('Tipo de terminación por episodio')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Gráfico de pastel (proporciones)
    plt.figure(figsize=(5,5))
    plt.pie(counts, labels=counts.index.str.capitalize(), autopct='%1.1f%%', startangle=140, colors=['#FF8888', '#8888FF', '#88FF88'])
    plt.title("Distribución de terminaciones de episodios")
    plt.tight_layout()
    plt.show()
    
    # Resumen final
    print("\nResumen:")
    print(f"• Porcentaje de llegadas: {porcentajes.get('llegada',0):.1f}%")
    print(f"• Porcentaje de choques: {porcentajes.get('choque',0):.1f}%")
    print(f"• Porcentaje de timeouts: {porcentajes.get('timeout',0):.1f}%")

    # Devuelve los datos si se quieren analizar más
    return df, counts, porcentajes

# Ejecución
if __name__ == "__main__":
    analizar_terminaciones("logs/terminaciones_0.5s.csv")
