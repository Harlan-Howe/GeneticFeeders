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

VOWELS = ["a","e","i","o","u","y"]
CONSONANTS = ["b","c","d","f","g","h","j","k","l","m","n","p","q","r","s","t","v","w","x","z"]
"""================================================================================================================
    Three methods for creating and altering names. Names in this program are of the format:
    consonant, vowel, consonant, vowel, consonant, vowel, consonant.
"""


def pick_name() -> str:
    """
    generates a random name
    :return: the name selected
    """
    return (f"{random.choice(CONSONANTS).upper()}{random.choice(VOWELS)}{random.choice(CONSONANTS)}"
            f"{random.choice(VOWELS)}{random.choice(CONSONANTS)}{random.choice(VOWELS)}{random.choice(CONSONANTS)}")


def mutate_name(name:str) -> str:
    """
    picks a new name that is one character different from the given name
    :param name: a source name, 7 characters long
    :return: a slightly different name, 7 characters long.
    """
    index = random.randint(0,6)

    if index == 0:
        new_name = random.choice(CONSONANTS).upper()
    else:
        new_name = name[0]

    for i in range(1,7):
        if index == i:
            if index%2==0:
                new_name += random.choice(CONSONANTS)
            else:
                new_name += random.choice(VOWELS)
        else:
            new_name += name[i]

    return new_name


def baby_name(name1:str, name2:str) -> str:
    """
    picks a new name, randomly composed of the letters in the parents' names
    :param name1: a 7-character string
    :param name2: another 7-character string
    :return: a new 7-character string.
    """
    baby_name = ""
    for i in range(7):
        if random.random() > 0.5:
            baby_name += name1[i]
        else:
            baby_name += name2[i]
    return baby_name

"""
==================================================================================================== FEEDER CLASS
"""
class Feeder:

    def __init__(self, genes: Optional[List[float]] = None):
        self.position: List[float] = [random.randint(0, 800), random.randint(0, 800)]
        self.orientation = random.random()*2*math.pi-math.pi
        self.speed = 15.0
        self.turn_ratio = 0.0  # a.k.a. angular velocity

        self.food_sensors = [0.0 for i in range(NUM_SENSORS)]   # detection levels of food and danger in various angles,
        self.danger_sensors = [0.0 for i in range(NUM_SENSORS)]  # ranging from -π to +π, relative to the orientation.

        self.color: Tuple[float, float, float] = (random.random() * 0.8, random.random() * 0.8 , random.random() * 0.8)

        # randomize genes or load them from "genes"
        if genes is None:
            self.genes = tuple([2*random.random()-1 for i in range(4*NUM_SENSORS)])
        else:
            self.genes = tuple(genes)

        self.is_alive = True
        self.food_level = 50
        self.age = 0.0
        self.death_reason = ""
        self.name = pick_name()


    def die(self, reason=""):
        """
        deactivate this feeder for the rest of this generation.
        :param reason: a short string (1 character?) explaining why this feeder died; e.g., "O" touched danger, "E" empty food.
        """
        self.is_alive = False
        self.death_reason = reason

    def rejuvenate(self):
        """
        reactivate this feeder for the next generation.
        """
        self.is_alive = True
        self.position = [random.randint(0, 800), random.randint(0, 800)]
        self.orientation = random.random() * 2 * math.pi - math.pi
        self.death_reason = ""
        self.food_level = 50
        self.speed = 15.0
        self.turn_ratio = 0.0
        self.age = 0.0

    def draw_self(self, canvas: np.ndarray, display_sensors=False):
        """
        draw this feeder in the simulation window.
        :param canvas: the window in which to draw
        :param display_sensors: whether to draw the lines representing when this feeder is sensing something.
        """
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
        cv2.putText(img=canvas, text=f"{self.name}",org=(int(self.position[0]-20),int(self.position[1]-15)),fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=self.color)
        health_color = (0,1,0)
        if self.food_level < 20:
            health_color = (0,0,1)
        cv2.line(img=canvas, pt1=(int(self.position[0]-20),int(self.position[1]-14)),
                 pt2=(int(self.position[0]-20+0.3*self.food_level),int(self.position[1]-14)),
                 color=health_color, thickness = 2)

    def clear_sensors(self):
        """
        reset the values of the sensors to zero, in preparation to start sensing again for this animation step.
        """
        self.food_sensors = [0.0 for _ in range(NUM_SENSORS)]
        self.danger_sensors = [0.0 for _ in range(NUM_SENSORS)]

    def detect(self, loc: Tuple[float, float] | List[float], isDanger=False):
        """
        update the sensors for this feeder, based on a piece of food or a danger at the given location.
        :param loc: the location of the food or danger
        :param isDanger: whether this is a danger object or a food object.
        """
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

    def animation_step(self, delta_t: float):
        """
        simulate one step of the feeder's life.
        :param delta_t: the time since the previous animation step
        """
        self.food_level -= CONSUMPTION_PER_SECOND*delta_t
        if self.food_level < 0:
            self.die()
            self.death_reason = "E"
            return

        self.age += delta_t

        self.update_feeder_motion_from_sensors()

        #  note: moves in the direction halfway between previous orientation and new orientation.
        self.orientation += self.turn_ratio*delta_t/2
        self.position = [self.position[0]+self.speed * delta_t * math.cos(self.orientation),
                         self.position[1]+self.speed * delta_t * math.sin(self.orientation)]
        self.orientation += self.turn_ratio * delta_t / 2


    def update_feeder_motion_from_sensors(self):
        """
        translates the values of the sensors to commands for the speed and turn ratio, based on the genes for this
        feeder. This is where the genes of the feeder have their effect; each one is multiplied by the corresponding
        sensor, and the totals are sent to the speed and turn_direction controls, both of which are capped.
        """
        for i in range(NUM_SENSORS):
            self.speed += (self.genes[i] * self.food_sensors[i] +
                           self.genes[i + NUM_SENSORS] * self.danger_sensors[i])
            self.turn_ratio += (self.genes[i + NUM_SENSORS * 2] * self.food_sensors[i] +
                                self.genes[i + NUM_SENSORS * 3] * self.danger_sensors[i])
        self.speed = min(MAX_SPEED, max(-MAX_SPEED, self.speed))
        self.turn_ratio = min(MAX_TURN_RATIO, max(-MAX_TURN_RATIO, self.turn_ratio))

    def __lt__(self, other: "Feeder") -> bool:
        """
        overrides the "<" operator, so that a list of feeders can be sorted.
        :param other: the feeder to compare to.
        :return: whether this feeder has a lower score than the other feeder.
        """
        if self.age == other.age:
            return self.food_level < other.food_level
        return self.age < other.age

    def __eq__(self, other: "Feeder") -> bool:
        """
        overrides the "==" operator, so that a list of feeders can be sorted.
        :param other: the feeder to compare this one to
        :return: whether these two feeders have equivalent scores
        """
        return self.age == other.age and self.food_level == other.food_level

    def display_attributes_at(self, canvas: np.ndarray, center: Tuple[int, int] | List[int], scale: float = 1.0):
        """
        draws a graphical representation of the genes for this feeder at the given location, as well as the name,
        and potentially the age and death reason if this feeder has died.
        :param canvas: the canvas to draw into, typically the stats window
        :param center: the location to draw this shape
        :param scale: a multiplier to the size of the shape we are drawing.
        :return:
        """
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
            cv2.putText(img=canvas, text=self.name, org= (int(center[0]-DANGER_SENSOR_RADIUS*scale),
                                                          int(center[1] - DANGER_SENSOR_RADIUS*scale - 30*scale)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=scale*2, color=self.color)

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
        baby.name = baby_name(self.name, other.name)
        return baby

    def get_mutated_version_of_Feeder(self) -> "Feeder":
        """
        "Yeah, there's a hole in the ozone layer above my head. So what?"

        Creates a new Feeder with a genetic code that may be slightly different from self's. I.e., there is some random
        chance that some random genes are changed by some random amount.

        :return: a new Feeder, a mutated version of self.
        """
        new_gene_set = list(copy.deepcopy(self.genes))

        # TODO: use random to potentially make one or more changes to these genes.
        new_Feeder = Feeder(new_gene_set)
        new_Feeder.name = mutate_name(self.name)
        return new_Feeder