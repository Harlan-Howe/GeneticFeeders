import copy
import math
import random
from typing import List, Tuple, Optional

import cv2
import numpy as np

MAX_SPEED = 30
MAX_TURN_RATIO = 0.2
FEEDER_RADIUS = 5
FOOD_SENSOR_RADIUS = 40
FOOD_SENSOR_RADIUS_SQUARED = math.pow(FOOD_SENSOR_RADIUS, 2)
DANGER_SENSOR_RADIUS = 100
DANGER_SENSOR_RADIUS_SQUARED = math.pow(DANGER_SENSOR_RADIUS, 2)
NUM_SENSORS = 16
CONSUMPTION_PER_SECOND = 4

class Feeder:

    def __init__(self, genes: Optional[List[float]] = None):
        self.position: List[float] = [random.randint(0, 800), random.randint(0, 800)]
        self.orientation = random.random()*2*math.pi-math.pi
        self.speed = 15.0
        self.turn_ratio = 0.0
        self.food_sensors = [0.0 for i in range(NUM_SENSORS)]
        self.danger_sensors = [0.0 for i in range(NUM_SENSORS)]

        self.color: Tuple[float, float, float] = (random.random() * 0.8, random.random() * 0.8 , random.random() * 0.8)
        if genes is None:
            self.genes = tuple([2*random.random()-1 for i in range(4*NUM_SENSORS)])
        else:
            self.genes = tuple(genes)
        self.is_alive = True
        self.food_level = 50
        self.age = 0.0
        self.death_reason = ""

    def reset(self):
        self.is_alive = True
        self.food_level = 50
        self.age = 0.0
        self.position: List[float] = [random.randint(0, 800), random.randint(0, 800)]
        self.orientation = random.random() * 2 * math.pi - math.pi
        self.speed = 15.0
        self.turn_ratio = 0.0
        self.death_reason = ""

    def die(self, reason=""):
        self.is_alive = False
        self.death_reason = reason

    def rejuvenate(self):
        self.is_alive = True
        self.position = [random.randint(0, 800), random.randint(0, 800)]
        self.orientation = random.random() * 2 * math.pi - math.pi
        self.death_reason = ""
        self.food_level = 50
        self.speed = 15.0
        self.turn_ratio = 0.0
        self.age = 0.0

    def draw_self(self, canvas: np.ndarray, display_sensors=False):
        if display_sensors:
            for i in range(NUM_SENSORS):
                angle = (i * math.pi * 2 / NUM_SENSORS + self.orientation) % (2*math.pi) - math.pi
                if self.danger_sensors[i]>0:
                    cv2.line(img=canvas, pt1=(int(self.position[0]), int(self.position[1])),
                             pt2=(int(self.position[0] + DANGER_SENSOR_RADIUS * math.cos(angle)),
                                  int(self.position[1] + DANGER_SENSOR_RADIUS * math.sin(angle))),
                             color=(0, 0.5 +self.danger_sensors[i]/2, 0),
                             thickness=1)
                if self.food_sensors[i]>0:
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
        cv2.putText(img=canvas, text=f"{self.age:3.2f}",org=(int(self.position[0]-10),int(self.position[1]-15)),fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=self.color)

    def clear_sensors(self):
        self.food_sensors = [0.0 for _ in range(NUM_SENSORS)]
        self.danger_sensors = [0.0 for _ in range(NUM_SENSORS)]

    def detect(self, loc: Tuple[float, float] | List[float], isDanger=False):
        threshold = FOOD_SENSOR_RADIUS
        threshold_squared = FOOD_SENSOR_RADIUS_SQUARED
        if isDanger:
            threshold = DANGER_SENSOR_RADIUS
            threshold_squared = DANGER_SENSOR_RADIUS_SQUARED
        distance_squared = math.pow(self.position[0] - loc[0], 2) + math.pow(self.position[1] - loc[1], 2)
        if distance_squared > threshold_squared:
            return

        distance = math.sqrt(distance_squared)
        proximity = 1.0 - distance/threshold

        theta = math.atan2(loc[1]-self.position[1], loc[0]-self.position[0])
        offset_diff = (theta - self.orientation + math.pi) % (2*math.pi)
        index = int((offset_diff / (2 * math.pi)) * NUM_SENSORS + 0.5) % NUM_SENSORS
        if isDanger:
            self.danger_sensors[index] = max(self.danger_sensors[index], proximity)
        else:
            self.food_sensors[index] = max(self.food_sensors[index],proximity)
        # print(f"{self.danger_sensors=}\t{self.food_sensors=}")

    def animation_step(self, delta_t:float):
        # self.speed = 0
        # self.turn_ratio = 0
        self.food_level -= CONSUMPTION_PER_SECOND*delta_t
        if self.food_level < 0:
            self.die()
            self.death_reason = "E"
            return
        self.age += delta_t
        for i in range(NUM_SENSORS):
            self.speed += (self.genes[i] * self.food_sensors[i] +
                           self.genes[i+NUM_SENSORS] * self.danger_sensors[i])
            self.turn_ratio += (self.genes[i+NUM_SENSORS*2] * self.food_sensors[i] +
                                self.genes[i+NUM_SENSORS*3] * self.danger_sensors[i])
        self.speed = min(MAX_SPEED, max(-MAX_SPEED, self.speed))
        self.turn_ratio = min(MAX_TURN_RATIO, max(-MAX_TURN_RATIO, self.turn_ratio))

        self.orientation += self.turn_ratio*delta_t/2
        self.position = [self.position[0]+self.speed * delta_t * math.cos(self.orientation),
                         self.position[1]+self.speed * delta_t * math.sin(self.orientation)]
        self.orientation += self.turn_ratio * delta_t / 2

        # print(f"{self.speed=}\t{self.turn_ratio=}")

    def __lt__(self, other):
        if self.age == other.age:
            return self.food_level < other.food_level
        return self.age < other.age

    def __eq__(self, other):

        return self.age == other.age and self.food_level == other.food_level

    def display_attributes_at(self, canvas:np.ndarray, center:Tuple[int,int]|List[int], scale:float = 1.0):
        angle_per_sensor = 360/NUM_SENSORS;
        for i in range(NUM_SENSORS):
            color_food_speed = (0, max(0, self.genes[i]), max(0, -self.genes[i]))
            color_danger_speed = (0, max(0, self.genes[i+NUM_SENSORS]), max(0, -self.genes[i+NUM_SENSORS]))
            color_food_turn = (0, max(0, self.genes[i+2*NUM_SENSORS]), max(0, -self.genes[i+2*NUM_SENSORS]))
            color_danger_turn = (0, max(0, self.genes[i + 3* NUM_SENSORS]), max(0, -self.genes[i + 3 * NUM_SENSORS]))

            cv2.ellipse(img=canvas,
                        center=center,
                        axes=(int(DANGER_SENSOR_RADIUS * scale * .55), int(DANGER_SENSOR_RADIUS * scale * .55)),
                        angle=0,
                        startAngle=(i - 0.45) * angle_per_sensor,
                        endAngle=(i + 0.45) * angle_per_sensor,
                        color=color_food_speed,
                        thickness=-1)
            cv2.ellipse(img=canvas,
                        center=center,
                        axes=(int(DANGER_SENSOR_RADIUS * scale * 0.7),int(DANGER_SENSOR_RADIUS * scale * 0.7)),
                        angle=0,
                        startAngle=(i-0.375)*angle_per_sensor,
                        endAngle=(i+0.375)*angle_per_sensor,
                        color = color_food_turn,
                        thickness = -1)

            cv2.ellipse(img=canvas,
                        center=center,
                        axes=(int(DANGER_SENSOR_RADIUS*scale*.85),int(DANGER_SENSOR_RADIUS*scale*.85)),
                        angle=0,
                        startAngle=(i-0.25)*angle_per_sensor,
                        endAngle=(i+0.25)*angle_per_sensor,
                        color = color_danger_speed,
                        thickness = -1)

            cv2.ellipse(img=canvas,
                        center=center,
                        axes=(int(DANGER_SENSOR_RADIUS*scale),int(DANGER_SENSOR_RADIUS*scale)),
                        angle=0,
                        startAngle=(i-0.125)*angle_per_sensor,
                        endAngle=(i+0.125)*angle_per_sensor,
                        color = color_danger_turn,
                        thickness = -1)

            cv2.circle(img=canvas, center=center, radius=int(FEEDER_RADIUS * 3.5 * scale),
                       color=(1, 1, 1),
                       thickness=-1)

            cv2.circle(img=canvas, center=center, radius=int(FEEDER_RADIUS * 3 * scale),
                       color=self.color,
                       thickness=-1)
            front = (int(center[0] + scale * 3 * FEEDER_RADIUS),
                     int(center[1]))
            cv2.line(img=canvas, pt1=center,
                     pt2=front,
                     color=(1.0, 1.0, 1.0),
                     thickness=3)

            cv2.line(img=canvas, pt1=center,
                     pt2=front,
                     color=(0.0, 0.0, 0.0),
                     thickness=1)

            if not self.is_alive:
                if self.food_level > 0:
                    cv2.putText(img=canvas, text=f"{self.age:3.2f} + {int(self.food_level)}",
                                org=(int(center[0] - DANGER_SENSOR_RADIUS*scale), int(center[1] - DANGER_SENSOR_RADIUS*scale)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=scale*2, color=self.color)
                else:
                    cv2.putText(img=canvas, text=f"{self.age:3.2f}   {self.death_reason}",
                                org=(int(center[0] - DANGER_SENSOR_RADIUS*scale), int(center[1] - DANGER_SENSOR_RADIUS*scale)),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=scale*2, color=self.color)

    def have_sex(self, other:"Feeder") -> "Feeder":
        """
        "When a mommy Feeder and a daddy Feeder really love each other...."

        Creates a new Feeder, based on random combination of the genetics of the parents.

        :param other: the mate to self
        :return: the baby feeder created by these two parents, self and other.
        """
        parent_1_genes = self.genes  # a list of 4 * NUM_SENSORS floats
        parent_2_genes = other.genes # another list of 4 * NUM_SENSORS floats.

        # make baby_genes become a new list of 4 * NUM_SENSORS floats.
        baby_genes = copy.deepcopy(parent_1_genes) # TODO: This is wrong. Do something sexier.

        baby = Feeder(genes=baby_genes)
        return baby

    def get_mutated_version(self) -> "Feeder":
        """
        creates a new Feeder with a genetic code that may be slightly different from self's. I.e., there is some random
        chance that some random genes are changed by some random amount.

        :return: a new Feeder, a mutated version of self.
        """
        new_gene_set = copy.deepcopy(self.genes)

        # TODO: use random to potentially make one or more changes to these genes.

        return Feeder(new_gene_set)