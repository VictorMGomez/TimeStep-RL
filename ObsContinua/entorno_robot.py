import time
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from laser_sensor import LaserSensor
from robot_diferencial import RobotDiferencial

class EntornoRobot(gym.Env):
    def __init__(self, num_obstacles=10):
        super().__init__()
        
        self.window = None
        self.clock = None
        self.render_mode = None
        self.current_time = 0.0  # Inicializar el tiempo del episodio

        # Medidas del entorno en metros
        self.WIDTH = 8.0  # Ancho de la ventana
        self.HEIGHT = 6.0  # Alto de la ventana
        self.BORDER = 0.1  # Marco de seguridad (obstáculo)
        
        # Escala para conversión a Pygame (ya definida en otro lugar)
        self.SCALE = 100  # Ejemplo: 1 metro = 100 píxeles

        # Espacio de acciones
        self.max_v = 0.2  # Velocidad lineal máxima (m/s)
        self.max_w = 2.84  # Velocidad angular máxima (rad/s)
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_w]),
            high=np.array([self.max_v, self.max_w]),
            dtype=np.float32
        )

        self.num_lasers = 32  # Número rayos láser
        self.step_count = 0
        # Inicializar robot diferencial
        self.robot = RobotDiferencial(
            r=0.033, d=0.16, tau=0.1, k=1.0, deadzone=0.01,
            x=self.WIDTH / 2, y=self.HEIGHT / 2, theta=0.0
        )
        
        # Inicializar sensor láser
        self.laser = LaserSensor(
            semiangle_beam=0.1, sigma_long=0.01, sigma_perp=0.01,
            max_z=3.0, start_angle=-np.pi/2, num_beams=self.num_lasers,
            angle_increment=np.pi / (self.num_lasers - 1)
        )
        
        max_dist = np.hypot(self.WIDTH, self.HEIGHT)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi] + [0.0] * self.num_lasers + [0.0, -self.max_w]),
            high=np.array([max_dist, np.pi] + [self.laser.max_z] * self.num_lasers + [self.max_v, self.max_w]),
            dtype=np.float32
        )

        # Definir el marco como una lista de listas con segmentos [(x1, y1, x2, y2)]
        self.marco = [
            [(self.BORDER, self.BORDER, self.WIDTH - self.BORDER, self.BORDER)],  # Borde superior
            [(self.WIDTH - self.BORDER, self.BORDER, self.WIDTH - self.BORDER, self.HEIGHT - self.BORDER)],  # Borde derecho
            [(self.WIDTH - self.BORDER, self.HEIGHT - self.BORDER, self.BORDER, self.HEIGHT - self.BORDER)],  # Borde inferior
            [(self.BORDER, self.HEIGHT - self.BORDER, self.BORDER, self.BORDER)]  # Borde izquierdo
        ]
        
        # Obstáculos aleatorios
        self.num_obstacles = num_obstacles  # Número de obstáculos
        self.obstacles = self._generar_obstaculos()
        
        # Parámetros de simulación
        self.dt = 0.01  # Paso de simulación en segundos
        self.target_position = self._generar_posicion_objetivo()
    
    def _generar_obstaculos(self):
        """Genera obstáculos aleatorios dentro del entorno."""
        obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                x = self.np_random.uniform(self.BORDER + 0.3, self.WIDTH - self.BORDER - 0.3)  # Asegurar margen
                y = self.np_random.uniform(self.BORDER + 0.3, self.HEIGHT - self.BORDER - 0.3)
                r = self.np_random.uniform(0.1, 0.3)

                # Verificar que el obstáculo no esté tocando el marco
                if self.BORDER < x - r and x + r < self.WIDTH - self.BORDER and \
                self.BORDER < y - r and y + r < self.HEIGHT - self.BORDER and all(np.hypot(x - ox, y - oy) > r + or_ + 0.7 for ox, oy, or_ in obstacles):
                    obstacles.append((x, y, r))
                    break  # Salir del loop si el obstáculo es válido
        return obstacles
        
    def _generar_posicion_objetivo(self):
        """Genera una posición aleatoria para el objetivo dentro del entorno."""
        while True:
            x = self.np_random.uniform(self.BORDER + 0.2, self.WIDTH - self.BORDER - 0.2)
            y = self.np_random.uniform(self.BORDER + 0.2, self.HEIGHT - self.BORDER - 0.2)

            # Asegurar que no está demasiado cerca de los obstáculos
            if all(np.hypot(x - ox, y - oy) > 0.5 + r for ox, oy, r in self.obstacles):
                return x, y  # Posición válida
    
    def reset(self, seed=None):
        """Reinicia el entorno colocando el robot y el objetivo en posiciones aleatorias."""
        super().reset(seed=seed)

        # Reiniciar contador de pasos
        self.step_count = 0

        #if self.render_mode:
        self.obstacles = self._generar_obstaculos()

        while True:
            x = self.np_random.uniform(self.BORDER + 0.5, self.WIDTH - self.BORDER - 0.5)
            y = self.np_random.uniform(self.BORDER + 0.5, self.HEIGHT - self.BORDER - 0.5)
            if all(np.hypot(x - ox, y - oy) > 0.5 for ox, oy, _ in self.obstacles):
                self.robot.x = x
                self.robot.y = y
                self.robot.theta = self.np_random.uniform(-np.pi, np.pi)
                break
        
        
        self.target_position = self._generar_posicion_objetivo()

        # Obtener observación inicial
        observation = self._obtener_observacion()
        return observation, {}
    
    def _obtener_observacion(self):
        """Obtiene la observación del estado actual del entorno."""
        dist = np.hypot(self.robot.x - self.target_position[0], self.robot.y - self.target_position[1])
        angle = np.arctan2(self.target_position[1] - self.robot.y, self.target_position[0] - self.robot.x) - self.robot.theta
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
        laser_distances, _ = self.laser.simulate_beam(self.robot.x, self.robot.y, self.robot.theta, self.obstacles, self.marco)
        laser_distances = laser_distances.tolist()

        v_real, w_real, _, _ = self.robot.calcular_velocidades()

        obs = [dist, angle] + laser_distances + [v_real, w_real]

        return np.array(obs, dtype=np.float32)

    def simulate_robot(self, v, w, duration):
        # Cálculo de potencias de rueda
        pot_l = (v - (w * self.robot.d / 2)) / self.robot.r
        pot_r = (v + (w * self.robot.d / 2)) / self.robot.r

        tiempo_simulado = 0.0
        render_interval = 1.0 / 10.0  # cada 0.1s simulados
        tiempo_ultimo_render = 0.0

        # Para control a tiempo real si estamos renderizando
        if self.render_mode:
            tiempo_mundo_real_inicio = time.perf_counter()

        # Bucle que avanza en trozos de self.dt, sin pasarse de 'duration'
        while tiempo_simulado < duration:
            # Ajustar el último paso para no exceder 'duration'
            dt_step = min(self.dt, duration - tiempo_simulado)

            # Integración de la dinámica
            self.robot.calcular_alpha(pot_l, pot_r, dt_step)
            v_real, w_real, _, _ = self.robot.calcular_velocidades()
            self.robot.actualizar_pose(v_real, w_real, dt_step)

            tiempo_simulado += dt_step

            # Render cada 0.1s simulados
            if self.render_mode and (tiempo_simulado - tiempo_ultimo_render) >= render_interval:
                self.render()
                tiempo_ultimo_render = tiempo_simulado

            # Throttle para no ir más rápido que el tiempo real
            if self.render_mode:
                elapsed_real = time.perf_counter() - tiempo_mundo_real_inicio
                sleep_time = float(tiempo_simulado - elapsed_real)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)


    def step(self, action):
        """Ejecuta un paso en la simulación asegurando que dure 1 segundo."""
        v, w = action
        start = time.perf_counter()

        self.simulate_robot(v, w, duration=2.0)

        self.step_count += 1
        max_steps = 75
        
        # Comprobar si hay choque
        laser_distances, _ = self.laser.simulate_beam(self.robot.x, self.robot.y, self.robot.theta, self.obstacles, self.marco)
        choque = any(d < 0.15 for d in laser_distances)
        
        # Calcular distancia actual al objetivo
        current_dist = np.hypot(self.robot.x - self.target_position[0], self.robot.y - self.target_position[1])
        llegada = current_dist < 0.3

        end = time.perf_counter()
        tiempo_real = end - start

        # Terminar el episodio si se alcanza el límite de pasos
        timeout = (self.step_count >= max_steps)
        terminated = choque or llegada
        truncated = timeout and not terminated
        

        recompensa = -100 if choque else 100 if llegada else 0.0
        info = {
            "tiempo_transcurrido": tiempo_real,
            "dt_elegido": 2.0,
            "truncated": truncated,
            "velocidad"    : v
        }

        return self._obtener_observacion(), recompensa, terminated, truncated, info

    def render(self, mode='human'):

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self._metros_pixeles(self.WIDTH), self._metros_pixeles(self.HEIGHT)))
            pygame.display.set_caption("Simulación Robot Diferencial con Gymnasium")
            self.clock = pygame.time.Clock()
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return  # Salir del render si la ventana se cierra

        self.window.fill((255, 255, 255))
        
        # Dibujar el marco del entorno
        pygame.draw.rect(self.window, (0, 0, 0), 
                     (self._metros_pixeles(self.BORDER), self._metros_pixeles(self.BORDER),
                      self._metros_pixeles(self.WIDTH - 2 * self.BORDER), self._metros_pixeles(self.HEIGHT - 2 * self.BORDER)), 3)

        # Simular láser
        distances, points = self.laser.simulate_beam(
            self.robot.x, self.robot.y, self.robot.theta, self.obstacles, self.marco
        )
        
        # Dibujar obstáculos
        for ox, oy, r in self.obstacles:
            pygame.draw.circle(self.window, (0, 0, 255), (self._metros_pixeles(ox), self._metros_pixeles(oy)), self._metros_pixeles(r), 2)
        
        # Dibujar el robot
        pygame.draw.circle(self.window, (0, 255, 0), (self._metros_pixeles(self.robot.x), self._metros_pixeles(self.robot.y)), self._metros_pixeles(0.1))
        
        # Dibujar orientación del robot
        end_x = self.robot.x + 0.3 * np.cos(self.robot.theta)
        end_y = self.robot.y + 0.3 * np.sin(self.robot.theta)
        pygame.draw.line(self.window, (0, 0, 0),
                         (self._metros_pixeles(self.robot.x), self._metros_pixeles(self.robot.y)),
                         (self._metros_pixeles(end_x), self._metros_pixeles(end_y)), 3)
        
        # Dibujar objetivo
        pygame.draw.circle(self.window, (255, 0, 0), (self._metros_pixeles(self.target_position[0]), self._metros_pixeles(self.target_position[1])), self._metros_pixeles(0.2))
        
        # 🕒 **Dibujar contador de tiempo**
        
        font = pygame.font.Font(None, 36)  # Fuente predeterminada, tamaño 36
        tiempo_texto = f"Tiempo: {self.current_time:.1f} s"  # Formato con 1 decimal
        text_surface = font.render(tiempo_texto, True, (0, 0, 0))  # Renderizar texto en negro
        self.window.blit(text_surface, (10, 10))  # Dibujar en la esquina superior izquierda
        
        # Dibujar los haces del láser
        """
        for distance, point in zip(distances, points):
            if point is not None:
                laser_end_x, laser_end_y = point
                pygame.draw.line(self.window, (255, 0, 0),
                                 (self._metros_pixeles(self.robot.x), self._metros_pixeles(self.robot.y)),
                                 (self._metros_pixeles(laser_end_x), self._metros_pixeles(laser_end_y)), 2)
                
                if distance < self.laser.max_z:
                    pygame.draw.circle(self.window, (0, 0, 0), (self._metros_pixeles(laser_end_x), self._metros_pixeles(laser_end_y)), 3)
        """            
        pygame.display.flip()
        self.clock.tick(30)

    def _metros_pixeles(self, metros):
        return int(metros * self.SCALE)