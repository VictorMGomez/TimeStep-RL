import time
import pygame
from entorno_robot import EntornoRobot

def main():
    pygame.init()
    env = EntornoRobot()
    env.render_mode = True 
    print("Midiendo simulate_robot(v=0.1, w=0.0, duration=dur)")
    for dur in [0.05, 0.06, 0.07, 0.1, 0.5, 1.0, 2.0, 3.0]:
        start = time.perf_counter()
        env.simulate_robot(v=0.1, w=0.0, duration=dur)
        elapsed = time.perf_counter() - start
        print(f"Solicitado {dur}s → transcurrido {elapsed:>6.3f}s (diff {elapsed - dur:>+.3f}s)")

    pygame.quit()

if __name__ == "__main__":
    main()