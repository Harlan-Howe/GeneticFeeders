import math
import random
from typing import Optional, List

import cv2
import numpy as np

DANGERBALL_RADIUS = 10

class DangerBall:

    def __init__(self, pos:Optional[List[int]] = None, vel: Optional[List[float]] = None):
        if pos is None:
            self.pos = [random.randint(0, 800), random.randint(0, 800)]
        else:
            self.pos = pos
        if vel is None:
            speed = random.random() * 10 + 10
            angle = random.random() * 2 * math.pi
            self.velocity = [speed * math.cos(angle), speed*math.sin(angle)]
        else:
            self.velocity = vel

    def animate_step(self, delta_t:float):
        if self.velocity[0] == 0.0 and self.velocity[1] == 0.0:
            return
        self.pos[0] += self.velocity[0] * delta_t
        self.pos[1] += self.velocity[1] * delta_t

        for i in range(2):
            if self.pos[i] < 0:
                self.pos[i] *= -1
                self.velocity[i] = abs(self.velocity[i])

            if self.pos[i] > 800:
                self.pos[i] = 1600 - self.pos[i]
                self.velocity[i] = - abs(self.velocity[i])

    def draw_self(self, canvas: np.ndarray):
        cv2.circle(img=canvas,
                   center=(int(self.pos[0]),int(self.pos[1])),
                   radius= DANGERBALL_RADIUS,
                   color=(0, 0, 0),
                   thickness=1)
