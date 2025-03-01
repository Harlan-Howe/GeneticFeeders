import random

import cv2
import numpy as np

FOOD_RADIUS = 4

class Food:

    def __init__(self):
        self.pos = (random.randint(FOOD_RADIUS,800-FOOD_RADIUS),
                    random.randint(FOOD_RADIUS,800-FOOD_RADIUS))

    def draw_self(self, canvas:np.ndarray):
        cv2.circle(img=canvas, center=self.pos, radius = FOOD_RADIUS, color=(0,0.5,0.25), thickness = -1)
