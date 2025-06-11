import numpy as np

class RobotDiferencial:
    def __init__(self, r, d, tau, k, deadzone, x, y, theta):
        self.r = r  # Radio de las ruedas
        self.d = d  # Distancia entre ruedas
        self.tau = tau  # Constante de tiempo del motor
        self.k = k  # Ganancia del motor
        self.deadzone = deadzone  # Zona muerta del motor
        self.alpha_l = 0
        self.alpha_r = 0
        self.x = x
        self.y = y
        self.theta = theta

    def calcular_alpha(self, pot_l, pot_r, delta_t):
        if delta_t <= 0:
            raise ValueError("El incremento de tiempo (delta_t) debe ser mayor que 0.")
        
        # Deadzone
        pot_l = 0 if abs(pot_l) <= self.deadzone else pot_l
        pot_r = 0 if abs(pot_r) <= self.deadzone else pot_r

        # Modelo de primer orden de la dinámica de las ruedas
        a = 1 / self.tau
        b = self.k / self.tau

        self.alpha_l = (self.alpha_l * (1 - delta_t * a) + delta_t * b * pot_l)
        self.alpha_r = (self.alpha_r * (1 - delta_t * a) + delta_t * b * pot_r)


    def calcular_velocidades(self):
        """Calcula las velocidades reales del robot basadas en las aceleraciones de las ruedas."""
        vl = self.r * self.alpha_l
        vr = self.r * self.alpha_r
        v = (vr + vl) / 2
        w = (vr - vl) / self.d
        #print(f"v: {v}, w: {w}, vl: {vl}, vr:{vr}")
        return v, w, vl, vr

    def actualizar_pose(self, v, w, delta_t):
        """Actualiza la posición y orientación del robot."""
        if delta_t <= 0:
            raise ValueError("El incremento de tiempo (delta_t) debe ser mayor que 0.")

        self.x += v * np.cos(self.theta) * delta_t
        self.y += v * np.sin(self.theta) * delta_t
        self.theta = (self.theta + w * delta_t + np.pi) % (2 * np.pi) - np.pi  # Mantener en [-pi, pi]

    def obtener_pose(self):
        """Retorna la posición y orientación actual del robot."""
        return self.x, self.y, self.theta
