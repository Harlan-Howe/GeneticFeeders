import math
import random
from typing import List, Tuple

import cv2
import numpy as np

MAX_VELOCITY = 30
MAX_TURN_RATIO = 0.2
FEEDER_RADIUS = 5
SENSOR_RADIUS = 40

class Feeder:

    def __init__(self):
        self.position: List[float] = [random.randint(0, 800), random.randint(0, 800)]
        self.orientation = 0.0
        self.velocity = 0.0
        self.turn_ratio = 0.0
        self.sensors = [0 for i in range(13)]
        self.color: Tuple[float, float, float] = (random.random(), random.random(), random.random())

    def draw_self(self, canvas: np.ndarray, display_sensors=False):
        cv2.circle(img=canvas, center=(int(self.position[0]), int(self.position[1])), radius=FEEDER_RADIUS, color=self.color,
                   thickness=-1)
        front = (int(self.position[0]+FEEDER_RADIUS*math.cos(self.orientation)),
                 int(self.position[1]+FEEDER_RADIUS*math.sin(self.orientation)))
        cv2.line(img=canvas,pt1=(int(self.position[0]), int(self.position[1])),
                 pt2=front,
                 color=(1.0, 1.0, 1.0),
                 thickness=3)

        cv2.line(img=canvas, pt1=(int(self.position[0]), int(self.position[1])),
                 pt2=front,
                 color=(0.0, 0.0, 0.0),
                 thickness=1)