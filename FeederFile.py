import math
import random
from typing import List, Tuple, Optional

import cv2
import numpy as np

MAX_SPEED = 30
MAX_TURN_RATIO = 0.2
FEEDER_RADIUS = 5
FOOD_SENSOR_RADIUS = 40
DANGER_SENSOR_RADIUS = 100
NUM_SENSORS = 16

class Feeder:

    def __init__(self, genes: Optional[Tuple[float, ...]] = None):
        self.position: List[float] = [random.randint(0, 800), random.randint(0, 800)]
        print(f"{self.position=}")
        self.orientation = random.random()*2*math.pi-math.pi
        self.speed = 0.0
        self.turn_ratio = 0.0
        self.food_sensors = [0.0 for i in range(NUM_SENSORS)]
        self.danger_sensors = [0.0 for i in range(NUM_SENSORS)]

        self.color: Tuple[float, float, float] = (random.random(), random.random(), random.random())
        if genes is None:
            self.genes = tuple([2*random.random()-1 for i in range(4*NUM_SENSORS)])
        else:
            self.genes = genes

    def draw_self(self, canvas: np.ndarray, display_sensors=False):
        if display_sensors:
            for i in range(NUM_SENSORS):
                angle = (i * math.pi * 2 / NUM_SENSORS + self.orientation) % (2*math.pi) - math.pi
                cv2.line(img=canvas, pt1=(int(self.position[0]), int(self.position[1])),
                         pt2=(int(self.position[0] + DANGER_SENSOR_RADIUS * math.cos(angle)),
                              int(self.position[1] + DANGER_SENSOR_RADIUS * math.sin(angle))),
                         color=(0.5, 0.5+self.danger_sensors[i]/2, 0.5),
                         thickness=1)
                cv2.line(img=canvas, pt1=(int(self.position[0]), int(self.position[1])),
                         pt2=(int(self.position[0] + FOOD_SENSOR_RADIUS * math.cos(angle)),
                              int(self.position[1] + FOOD_SENSOR_RADIUS * math.sin(angle))),
                         color=(0.0, 0.0, self.food_sensors[i]),
                         thickness=1)

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

    def detect(self, loc: Tuple[float, float]|List[float]):
        theta = math.atan2(loc[1]-self.position[1], loc[0]-self.position[0])
        diff = theta - self.orientation
        diff = (diff + math.pi) % (2*math.pi)
        self.food_sensors[int((diff / (2 * math.pi)) * NUM_SENSORS + 0.5) % NUM_SENSORS] = 1
        # print(f"{theta=}\t{self.orientation=}\t{diff=}\t{self.food_sensors}")

    def animation_step(self, delta_t:float):
        self.speed = 0
        self.turn_ratio = 0
        for i in range(NUM_SENSORS):
            self.speed += (self.genes[i] * self.food_sensors[i] +
                           self.genes[i+NUM_SENSORS] * self.danger_sensors[i])
            self.turn_ratio += (self.genes[i+NUM_SENSORS*2] * self.food_sensors[i] +
                                self.genes[i+NUM_SENSORS*3] * self.danger_sensors[i])
        self.speed = max(MAX_SPEED, min(-MAX_SPEED, self.speed))
        self.turn_ratio = max(MAX_TURN_RATIO, min(-MAX_TURN_RATIO, self.turn_ratio))

        self.orientation += self.turn_ratio*delta_t/2
        self.position = [self.position[0]+self.speed*delta_t*math.cos(self.orientation),
                         self.position[1]+self.speed * delta_t * math.sin(self.orientation)]
        self.orientation += self.turn_ratio * delta_t / 2




