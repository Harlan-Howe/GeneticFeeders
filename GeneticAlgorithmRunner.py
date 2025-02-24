import math
from datetime import datetime
from typing import List

import cv2
import numpy as np

from DangerBallFile import DangerBall, DANGERBALL_RADIUS
from FeederFile import Feeder, FEEDER_RADIUS
from FoodFile import Food, FOOD_RADIUS

MAX_CYCLE_DURATION = 60
FOOD_THRESHOLD_SQUARED = math.pow(FOOD_RADIUS + FEEDER_RADIUS, 2)
DANGER_THRESHOLD_SQUARED = math.pow(DANGERBALL_RADIUS + FEEDER_RADIUS, 2)

class GeneticAlgorithmRunner:

    def __init__(self):
        self.main_canvas = np.ones((800, 800, 3), dtype=float)
        self.stats_canvas = np.ones((600, 600, 3), dtype=float)
        cv2.imshow("stats", self.stats_canvas)
        cv2.moveWindow("stats", 800, 0)

        self.moving_danger_list: List[DangerBall] = []
        self.all_dangers: List[DangerBall] = []
        self.food_list: List[Food] = []
        self.feeder_list: List[Feeder] = []
        self.reset_feeder_list()
        self.create_dangers_and_food()

        self.latest = datetime.now()

        self.cycle_ongoing = True
        self.age_of_cycle = 0.0

    def create_dangers_and_food(self):
        self.create_moving_dangers()
        self.create_danger_walls()
        self.create_food()

    def create_moving_dangers(self):
        for i in range(30):
            db = DangerBall()
            self.moving_danger_list.append(db)
            self.all_dangers.append(db)

    def create_danger_walls(self):
        for i in range(int(800 / DANGERBALL_RADIUS / 2 + 1)):
            self.all_dangers.append(
                DangerBall(pos=[int(-DANGERBALL_RADIUS / 2 + i * DANGERBALL_RADIUS * 2), int(-DANGERBALL_RADIUS / 2)],
                           vel=[0, 0]))
            self.all_dangers.append(
                DangerBall(pos=[int(-DANGERBALL_RADIUS / 2), int(-DANGERBALL_RADIUS / 2 + i * DANGERBALL_RADIUS * 2)],
                           vel=[0, 0]))
            self.all_dangers.append(DangerBall(
                pos=[int(-DANGERBALL_RADIUS / 2 + i * DANGERBALL_RADIUS * 2), int(800 + DANGERBALL_RADIUS / 2)],
                vel=[0, 0]))
            self.all_dangers.append(DangerBall(
                pos=[int(800 + DANGERBALL_RADIUS / 2), int(-DANGERBALL_RADIUS / 2 + i * DANGERBALL_RADIUS * 2)],
                vel=[0, 0]))

    def create_food(self):
        for i in range(200):
            self.food_list.append(Food())

    def reset_feeder_list(self):
        self.feeder_list.clear()
        for i in range(25):
            self.feeder_list.append(Feeder())
        self.cycle_ongoing = True
        self.age_of_cycle = 0.0

    def display_feeders(self, canvas: np.ndarray):
        for i in range(5):
            for j in range(5):
                self.feeder_list[i * 5 + j].display_attributes_at(canvas, (110 * j + 60, 110 * i + 60), 0.5)

    def animation_loop(self):
        while True:
            now = datetime.now()
            delta_t = (now - self.latest).total_seconds()
            self.age_of_cycle += delta_t
            self.latest = now

            main_canvas = np.ones((800, 800, 3), dtype=float)

            self.clear_all_live_feeder_sensors()
            self.move_and_draw_dangers(delta_t, main_canvas)
            self.detect_all_dangers()
            self.detect_all_food()
            self.move_all_feeders(delta_t)
            self.check_for_eaten_food()
            self.check_for_feeder_danger_collisions()
            self.draw_all_food(main_canvas)
            self.update_stats_window()
            self.draw_all_feeders(main_canvas)

            if self.cycle_ongoing and self.age_of_cycle >= MAX_CYCLE_DURATION:
                self.kill_all_feeders()

            cv2.putText(img=main_canvas, text=f"Time: {self.age_of_cycle}", org = (700,775),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color = (0,0,0))
            cv2.imshow("Canvas", main_canvas)
            response = cv2.waitKey(10)
            if response != -1:
                self.reset_feeder_list()

    def kill_all_feeders(self):
        for bug in self.feeder_list:
            bug.die()
        self.cycle_ongoing = False

    def draw_all_feeders(self, main_canvas):
        live_feeders = 0
        for bug in self.feeder_list:
            if bug.is_alive:
                bug.draw_self(canvas=main_canvas, display_sensors=True)
                live_feeders += 1
        cv2.putText(img=main_canvas, text=f"Num bugs: {live_feeders}", org=(10, 775), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(0, 0, 0))
        if live_feeders == 0:
            self.cycle_ongoing = False

    def update_stats_window(self):
        if self.cycle_ongoing:
            self.feeder_list.sort(reverse=True)
            stats_canvas = np.ones((600, 600, 3), dtype=float)
            self.display_feeders(stats_canvas)
            cv2.imshow("stats", stats_canvas)

    def draw_all_food(self, main_canvas):
        for f in self.food_list:
            f.draw_self(canvas=main_canvas)

    def check_for_feeder_danger_collisions(self):
        for db in self.all_dangers:
            for bug in self.feeder_list:
                if bug.is_alive:
                    if math.pow(db.pos[0] - bug.position[0], 2) + math.pow(db.pos[1] - bug.position[1],
                                                                           2) < DANGER_THRESHOLD_SQUARED:
                        bug.die()

    def check_for_eaten_food(self):
        eaten_food_list: List[Food] = []
        for f in self.food_list:
            for bug in self.feeder_list:
                if bug.is_alive:
                    if math.pow(f.pos[0] - bug.position[0], 2) + math.pow(f.pos[1] - bug.position[1],
                                                                          2) < FOOD_THRESHOLD_SQUARED:
                        bug.food_level = min(100, bug.food_level + 10)
                        eaten_food_list.append(f)
        for ef in eaten_food_list:
            self.food_list.remove(ef)
            self.food_list.append(Food())

    def move_all_feeders(self, delta_t):
        for bug in self.feeder_list:
            if bug.is_alive:
                bug.animation_step(delta_t)

    def detect_all_food(self):
        for f in self.food_list:
            for bug in self.feeder_list:
                if bug.is_alive:
                    bug.detect(f.pos, False)

    def detect_all_dangers(self):
        for db in self.all_dangers:
            for bug in self.feeder_list:
                if bug.is_alive:
                    bug.detect(db.pos, True)

    def move_and_draw_dangers(self, delta_t, main_canvas):
        for db in self.moving_danger_list:
            db.animate_step(delta_t)
            db.draw_self(main_canvas)

    def clear_all_live_feeder_sensors(self):
        for bug in self.feeder_list:
            if bug.is_alive:
                bug.clear_sensors()


if __name__ == "__main__":
    gar = GeneticAlgorithmRunner()
    gar.animation_loop()
    cv2.destroyAllWindows()