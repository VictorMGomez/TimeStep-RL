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
        self.current_time = 0.0

        # Dimensiones del entorno (m)
        self.WIDTH = 8.0
        self.HEIGHT = 6.0
        self.BORDER = 0.1
        self.SCALE = 100  # 1m = 100px

        # --- Espacio de acciones ---
        self.max_v = 0.2       # m/s
        self.max_w = 2.84     # rad/s
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_w], dtype=np.float32),
            high=np.array([self.max_v,  self.max_w], dtype=np.float32),
            dtype=np.float32
        )

        # --- Discretización de observaciones ---
        self.num_dist_categories  = 5
        self.num_angle_categories = 8
        self.num_laser_categories = 5
        self.num_lasers          = 32

        self.observation_space = spaces.MultiDiscrete(
            [self.num_dist_categories,
             self.num_angle_categories] +
             [self.num_laser_categories]*self.num_lasers
        )

        # Robot y láser
        self.robot = RobotDiferencial(
            r=0.033, d=0.16, tau=0.1, k=1.0, deadzone=0.01,
            x=self.WIDTH/2, y=self.HEIGHT/2, theta=0.0
        )
        self.laser = LaserSensor(
            semiangle_beam=0.1, sigma_long=0.01, sigma_perp=0.01,
            max_z=3.0, start_angle=-np.pi/2,
            num_beams=self.num_lasers,
            angle_increment=np.pi/(self.num_lasers-1)
        )

        # Contorno (polígonos) para colisiones
        self.marco = [
            [(self.BORDER, self.BORDER,
              self.WIDTH - self.BORDER, self.BORDER)],
            [(self.WIDTH - self.BORDER, self.BORDER,
              self.WIDTH - self.BORDER, self.HEIGHT - self.BORDER)],
            [(self.WIDTH - self.BORDER, self.HEIGHT - self.BORDER,
              self.BORDER, self.HEIGHT - self.BORDER)],
            [(self.BORDER, self.HEIGHT - self.BORDER,
              self.BORDER, self.BORDER)],
        ]

        # Obstáculos
        self.num_obstacles = num_obstacles
        self.obstacles     = self._generar_obstaculos()

        # Simulación
        self.dt = 0.01     # paso interno (s)
        self.target_position = self._generar_posicion_objetivo()
        self.step_count     = 0

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
        super().reset(seed=seed)

        self.step_count = 0
        
        self.obstacles = self._generar_obstaculos()

        while True:
            x = np.random.uniform(self.BORDER+0.2, self.WIDTH-self.BORDER-0.2)
            y = np.random.uniform(self.BORDER+0.2, self.HEIGHT-self.BORDER-0.2)
            if all(np.hypot(x-ox, y-oy)>0.5
                   for ox,oy,_ in self.obstacles):
                self.robot.x     = x
                self.robot.y     = y
                self.robot.theta = np.random.uniform(-np.pi, np.pi)
                break
        self.target_position = self._generar_posicion_objetivo()
        return self._obtener_observacion(), {}

    def _obtener_observacion(self):
        # distancia y ángulo al objetivo
        dx, dy = (self.target_position[0]-self.robot.x,
                  self.target_position[1]-self.robot.y)
        dist  = np.hypot(dx, dy)
        angle = (np.arctan2(dy, dx) - self.robot.theta + np.pi)%(2*np.pi) - np.pi

        laser_distances, _ = self.laser.simulate_beam(
            self.robot.x, self.robot.y, self.robot.theta,
            self.obstacles, self.marco
        )
        return self._discretizar_observaciones(dist, angle, laser_distances)

    def _discretizar_observaciones(self, dist, angle, laser_distances):
        # distancia
        d_bin = min(int(dist/(np.hypot(self.WIDTH,self.HEIGHT)/self.num_dist_categories)),
                    self.num_dist_categories-1)
        # ángulo
        a_bin = min(int((angle+np.pi)/(2*np.pi)*self.num_angle_categories),
                    self.num_angle_categories-1)
        # láseres
        l_bins = []
        for d in laser_distances:
            idx = min(int(d/self.laser.max_z*self.num_laser_categories),
                      self.num_laser_categories-1)
            l_bins.append(idx)
        return np.array([d_bin, a_bin] + l_bins, dtype=np.int32)

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

        self.simulate_robot(v, w, duration=0.5)

        self.step_count += 1
        max_steps = 200
        
        # Comprobar si hay choque
        laser_distances, _ = self.laser.simulate_beam(self.robot.x, self.robot.y, self.robot.theta, self.obstacles, self.marco)
        choque = any(d < 0.15 for d in laser_distances)
        
        # Calcular distancia actual al objetivo
        current_dist = np.hypot(self.robot.x - self.target_position[0], self.robot.y - self.target_position[1])
        llegada = current_dist < 0.3

        # Terminar el episodio si se alcanza el límite de pasos
        timeout = (self.step_count >= max_steps)
        terminated = choque or llegada
        truncated = timeout and not terminated

        recompensa = -100 if choque else 100 if llegada else 0.0
        info = {
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
        """
        font = pygame.font.Font(None, 36)  # Fuente predeterminada, tamaño 36
        tiempo_texto = f"Tiempo: {self.current_time:.1f} s"  # Formato con 1 decimal
        text_surface = font.render(tiempo_texto, True, (0, 0, 0))  # Renderizar texto en negro
        self.window.blit(text_surface, (10, 10))  # Dibujar en la esquina superior izquierda
        """
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

    def _metros_pixeles(self, m):
        return int(m*self.SCALE)
