import math
import random
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
NUM_FEEDERS = 25

class GeneticAlgorithmRunner:

    def __init__(self):
        self.main_canvas = np.ones((800, 800, 3), dtype=float)
        self.stats_canvas = np.ones((600, 600, 3), dtype=float)
        cv2.imshow("stats", self.stats_canvas)
        cv2.moveWindow("stats", 800, 100)

        self.moving_danger_list: List[DangerBall] = []
        self.all_dangers: List[DangerBall] = []
        self.food_list: List[Food] = []
        self.feeder_list: List[Feeder] = []
        self.reset_feeder_list()
        self.create_dangers_and_food()

        self.latest = datetime.now()

        self.cycle_ongoing = True
        self.age_of_cycle = 0.0
        self.generation_number = 0
        self.should_save_this_generation = False
        self.program_run_number = random.randint(1000,9999)

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

    def reset_feeder_list(self, all_weights:List[List[float]] = None, names:List[str] = None):
        self.feeder_list.clear()
        for i in range(NUM_FEEDERS):
            if all_weights is None:
                self.feeder_list.append(Feeder())
            else:
                self.feeder_list.append(Feeder(genes=all_weights[i]))
                self.feeder_list[i].name = names[i]
        self.cycle_ongoing = True
        self.age_of_cycle = 0.0
        self.should_save_this_generation = False


    def display_feeders(self, canvas: np.ndarray):

        num_rows = int(math.sqrt(len(self.feeder_list)))
        num_cols = math.ceil(len(self.feeder_list)/num_rows)
        for i in range(num_rows):
            for j in range(num_cols):
                self.feeder_list[i * num_cols + j].display_attributes_at(canvas, (110 * j + 60, 130 * i + 90), 0.5)

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
            if self.cycle_ongoing and self.age_of_cycle >= MAX_CYCLE_DURATION:
                self.kill_all_feeders()
            self.update_stats_window()

            if self.cycle_ongoing:
                cv2.putText(img=main_canvas, text=f"Time: {self.age_of_cycle:3.2f}", org = (700,775),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color = (0,0,0))
            cv2.putText(img=main_canvas, text=f"Generation {self.generation_number}",org=(10,10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color = (0,0,0))
            cv2.putText(img=main_canvas, text=f"run #: {self.program_run_number}", org = (700,10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,0))
            self.draw_all_feeders(main_canvas)
            cv2.imshow("Canvas", main_canvas)
            response = cv2.waitKey(10)
            if response == 115: #  ascii for s
               self.should_save_this_generation = True

            if not self.cycle_ongoing:
                if self.should_save_this_generation:
                    self.save_generation(f"{self.save_filename}-{self.generation_number}")
                self.advance_generation()

                self.cycle_ongoing = True
                self.age_of_cycle = 0.0
                self.should_save_this_generation = False
                self.generation_number += 1

    def kill_all_feeders(self):
        for bug in self.feeder_list:
            bug.die()


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
            stats_canvas = np.ones((700, 600, 3), dtype=float)
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
                        bug.food_level = 0
                        bug.death_reason = "O"

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

    def initial_setup(self):
        load_YN = input("Do you want to load an existing generation? (Y/N) ").lower()
        if load_YN == 'y':
            load_filename = input("Enter the name of the file, or type 'cancel' to change your mind. ")
            if load_filename != "cancel":
                self.load_generation(filename=load_filename)

        self.save_filename = f"generation {self.program_run_number}"
        print("Press 's' to save the current generation at the end of a cycle.")

    def save_generation(self, filename):
        text_to_write = f"{self.program_run_number}\n{self.generation_number}\n"
        for bug in self.feeder_list:
            text_to_write += bug.name
            for i in range(len(bug.genes)):
                text_to_write += (f"\t{bug.genes[i]}")
            text_to_write += "\n"
        try:
            with open(filename, "w") as file:
                file.write(text_to_write)
            print(f"Successfully wrote to {filename}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def load_generation(self, filename):
        all_weights:List[List[float]] = []
        try:
            with open(filename, "r") as file:
                self.program_run_number = int(file.readline())
                self.generation_number = int(file.readline())
                names: List[str] = []
                line = file.readline()
                while line:
                    parts = line.split("\t")
                    names.append(parts[0])
                    del(parts[0])
                    weights = []
                    for weight_string in parts:
                        weights.append(float(weight_string))
                    all_weights.append(weights)
                    line = file.readline()
            self.reset_feeder_list(all_weights, names)
        except Exception as e:
            print(f"Problem opening file: {e}")

    def advance_generation(self):
        # Dummy behavior. Just rejuvenates every Feeder, so the next generation is the same as this one.
        for bug in self.feeder_list:
            bug.rejuvenate()

if __name__ == "__main__":
    gar = GeneticAlgorithmRunner()
    gar.initial_setup()
    gar.animation_loop()
    cv2.destroyAllWindows()