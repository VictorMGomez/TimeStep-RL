Descripción del Proyecto
Este proyecto explora el aprendizaje por refuerzo en robótica móvil, centrándose en el impacto de la duración de las acciones (dt) sobre el rendimiento y el aprendizaje de un robot diferencial simulado. Utilizando Gymnasium y Stable-Baselines3, se han desarrollado varios entornos personalizados en Python que simulan escenarios con obstáculos, sensores láser y objetivos aleatorios.
El repositorio incluye tres variantes principales de entorno:

  ObsContinua: Observaciones continuas.

  ObsDiscreta: Observaciones discretas, pensadas para experimentos sin normalización.

  TiempoVariable: Permite que el agente elija dinámicamente la duración de cada acción (dt), investigando así la relación entre la frecuencia de control y el aprendizaje.

El objetivo principal es analizar y comparar cómo afecta el intervalo de tiempo de las acciones al comportamiento aprendido y al desempeño general del agente robótico, proporcionando un entorno flexible y reproducible para experimentación en simulación.
