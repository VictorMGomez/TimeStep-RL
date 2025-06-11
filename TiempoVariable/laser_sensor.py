import numpy as np

class LaserSensor:
    def __init__(self, semiangle_beam=0, sigma_long=0, sigma_perp=0, max_z=10, 
                 start_angle=0, num_beams=1, angle_increment=0):
        self.semiangle_beam = semiangle_beam
        self.sigma_long = sigma_long
        self.sigma_perp = sigma_perp
        self.max_z = max_z
        self.start_angle = start_angle
        self.num_beams = num_beams
        self.angle_increment = angle_increment

    def simulate_beam(self, x_beam, y_beam, theta_beam, map_circles, map_polygons):
        distances = []
        points = []

        for i in range(self.num_beams):
            beam_angle = theta_beam + self.start_angle + i * self.angle_increment + np.random.uniform(-self.semiangle_beam, self.semiangle_beam)
            distance, obstacle_x, obstacle_y = simulate_range_beam(
                x_beam, y_beam, beam_angle, map_circles, map_polygons,
                self.semiangle_beam, self.sigma_long, self.sigma_perp, self.max_z
            )
            distances.append(distance)
            points.append((obstacle_x, obstacle_y))

        return np.array(distances, dtype=np.float32), points


def simulate_range_beam(x_beam, y_beam, theta_beam, map_circles, map_polygons, semiangle_beam, sigma_long, sigma_perp, max_z):
    z = max_z
    closest_point = (x_beam + max_z * np.cos(theta_beam), 
                     y_beam + max_z * np.sin(theta_beam))

    found_intersection = False

    # Detectar intersecciones con círculos
    for circle in map_circles:
        x_c, y_c, r = circle
        lambda_c = ray_circle_cut(x_beam, y_beam, theta_beam, x_c, y_c, r)

        if lambda_c is not None and lambda_c < z:
            z = lambda_c
            found_intersection = True
            closest_point = (
                x_beam + lambda_c * np.cos(theta_beam),
                y_beam + lambda_c * np.sin(theta_beam)
            )

    # Detectar intersecciones con polígonos
    for polygon in map_polygons:
        num_vertices = len(polygon)
        for i in range(num_vertices):
            x1, y1, x2, y2 = polygon[i]
            theta_ray = theta_beam + np.random.normal(0, sigma_perp) if sigma_perp > 0 else theta_beam
            _, _, lambda_m = ray_segment_cut(x_beam, y_beam, theta_ray, x1, y1, x2, y2)

            if not np.isinf(lambda_m) and lambda_m < z:
                z = lambda_m
                found_intersection = True
                closest_point = (
                    x_beam + lambda_m * np.cos(theta_ray),
                    y_beam + lambda_m * np.sin(theta_ray)
                )

    return z, closest_point[0], closest_point[1]


def ray_circle_cut(x_ray, y_ray, theta_ray, x_c, y_c, r):
    dx = np.cos(theta_ray)
    dy = np.sin(theta_ray)
    fx = x_ray - x_c
    fy = y_ray - y_c
    
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - r**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    
    if t1 > 0:
        return t1
    elif t2 > 0:
        return t2
    else:
        return None


def ray_segment_cut(x_ray, y_ray, theta_ray, x1, y1, x2, y2):
    """
    Calcula la intersección de un rayo con un segmento.
    
    Parámetros:
    - x_ray, y_ray: Origen del rayo.
    - theta_ray: Ángulo del rayo en radianes.
    - x1, y1, x2, y2: Coordenadas del segmento.
    
    Retorna:
    - bool: Si hay intersección dentro del segmento.
    - int: Posición relativa (-1: antes, 0: dentro, 1: después).
    - float: Distancia a la intersección (np.inf si no hay intersección).
    """
    # Dirección del rayo
    u_x = np.cos(theta_ray)
    u_y = np.sin(theta_ray)
    
    # Vector del segmento
    w_x = x2 - x1
    w_y = y2 - y1

    A = np.array([[u_x, -w_x], [u_y, -w_y]])
    
    if np.linalg.matrix_rank(A) < 2:

        return False, 0, np.inf 
    
    E = np.array([x1 - x_ray, y1 - y_ray])
    lambda_ray, beta = np.linalg.solve(A, E)

    # Comprobar si la intersección ocurre en la dirección del rayo
    if lambda_ray < 0:
        return False, 0, np.inf  # Intersección detrás del rayo

    if 0 <= beta <= 1:
        return True, 0, lambda_ray
    elif beta < 0:
        return False, -1, np.inf  # Intersección antes del segmento
    else:
        return False, 1, np.inf  # Intersección después del segmento
