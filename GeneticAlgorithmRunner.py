import math
import random
from datetime import datetime
from typing import List, Set

import cv2
import numpy as np

from DangerBallFile import DangerBall, DANGERBALL_RADIUS
from FeederFile import Feeder, FEEDER_RADIUS
from FoodFile import Food, FOOD_RADIUS



DISPLAY_SENSORS = False
GRAPHIC_SIMULATION = True
DISPLAY_GRAPH = False

MAX_CYCLE_DURATION = 60
FOOD_THRESHOLD_SQUARED = math.pow(FOOD_RADIUS + FEEDER_RADIUS, 2)
DANGER_THRESHOLD_SQUARED = math.pow(DANGERBALL_RADIUS + FEEDER_RADIUS, 2)
NUM_FEEDERS = 81
NUM_MOVING_DANGERS = 30
NUM_FOOD = 200

GRAPH_SIZE = 400
GRAPH_MARGIN = 20

class GeneticAlgorithmRunner:

    def __init__(self):
        self.program_run_number = random.randint(1000, 9999)  # a random 4-digit id for this run.
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

        self.cycle_ongoing = True
        self.age_of_cycle = 0.0
        self.generation_number = 0
        self.should_save_this_generation = False
        self.live_feeders = NUM_FEEDERS

        #  stuff for statistics
        self.best_score_per_generation: List[float] = []
        self.mean_score_per_generation: List[float] = []

    def create_dangers_and_food(self):
        self.create_moving_dangers()
        self.create_danger_walls()
        self.create_food()

    def create_moving_dangers(self):
        """
        creates the circles that move around the canvas, deadly to the feeders.
        """
        for i in range(NUM_MOVING_DANGERS):
            db = DangerBall()
            self.moving_danger_list.append(db)
            self.all_dangers.append(db)

    def create_danger_walls(self):
        """
        creates the circles that represent the border of the canvas. These are just outside the visible canvas and do
        not move. They, too, are deadly to the feeders.
        """
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
        """
        creates a random selection of food items on the canvas, the green solid dots.
        """
        for i in range(NUM_FOOD):
            self.food_list.append(Food())

    def reset_feeder_list(self, all_weights:List[List[float]] = None, names:List[str] = None):
        """
        generates a new set of feeders. If all_weights is None, then they are generated randomly. Otherwise, they
        are generated based on the weights in all_weights, and use the names given
        :param all_weights: a List of weight lists to populate the feeders.
        :param names: the names that should be given to the feeders.
        """
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
        """
        Tells all the feeders to draw their attributes in the stats window, in a grid.
        :param canvas: the stats window in which to draw.
        """
        cv2.putText(img=canvas, text=f"Generation: {self.generation_number}", org=(10,10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(0, 0, 0))
        num_rows = int(math.sqrt(len(self.feeder_list)))
        num_cols = math.ceil(len(self.feeder_list)/num_rows)
        scale = 2.5/num_cols
        feeder_width = int(600/num_cols)
        for i in range(num_rows):
            for j in range(num_cols):
                self.feeder_list[i * num_cols + j].display_attributes_at(canvas, (feeder_width * j + 60, (feeder_width+10) * i + 90), scale)

    def animation_loop(self):
        """
        the primary "game loop" that makes the feeders and dangers move around and interact with each other and the food.
        """
        self.latest = datetime.now()
        while True:
            now = datetime.now()
            delta_t = (now - self.latest).total_seconds()
            if not GRAPHIC_SIMULATION:
                delta_t *= 10
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
            if GRAPHIC_SIMULATION:
                self.draw_all_food(main_canvas)
            if self.cycle_ongoing and self.age_of_cycle >= MAX_CYCLE_DURATION:
                self.kill_all_feeders()
            self.count_live_feeders()
            if GRAPHIC_SIMULATION:
                self.update_stats_window()

                self.draw_labels_in_simulation_window(main_canvas)
                self.draw_all_feeders(main_canvas)
                cv2.imshow("Canvas", main_canvas)
                response = cv2.waitKey(10)
            else:
                response = cv2.waitKey(1)
            if response == 115 or response == 83: #  ascii for s or S -- for Save
               self.should_save_this_generation = True

            if response == 113 or response == 81:  # ascii for q or Q  -- for Quit!
                break

            if not self.cycle_ongoing:
                self.handle_end_of_generation()



    def draw_labels_in_simulation_window(self, main_canvas):
        """
        draws the information in the four corners of the simulation canvas
        :param main_canvas: the canvas that displays the simulation
        """
        if self.cycle_ongoing:
            cv2.putText(img=main_canvas, text=f"Time: {self.age_of_cycle:3.2f}", org=(700, 775),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0))
            cv2.putText(img=main_canvas, text=f"Num feeders: {self.live_feeders}", org=(10, 775),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=(0, 0, 0))
        cv2.putText(img=main_canvas, text=f"Generation {self.generation_number}", org=(10, 10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0))
        cv2.putText(img=main_canvas, text=f"run #: {self.program_run_number}", org=(700, 10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0))

    def handle_end_of_generation(self):
        """
        takes care of the legwork when all the feeders have died (or have been killed) to analyze the results of this
        generation and reset for the next generation.
        """
        self.calculate_stats_for_generation()

        if self.should_save_this_generation:
            self.save_generation(f"{self.save_filename}-{self.generation_number}.dat")
        self.cycle_ongoing = True
        self.update_stats_window()
        cv2.waitKey(10)

        self.advance_generation()

        self.age_of_cycle = 0.0
        self.should_save_this_generation = False  # reset "s" key.
        self.generation_number += 1

    def calculate_stats_for_generation(self):
        """
        Now that generation is over, computes the score for the best feeder and the mean of the scores for all the
        feeders for this generation and appends them to the running statistics for the various generations.
        """
        self.feeder_list.sort(reverse=True)
        total_score = 0
        for feeder in self.feeder_list:
            if feeder.age >= MAX_CYCLE_DURATION:
                total_score += (100 + feeder.food_level)
            else:
                total_score += 100 * feeder.age / MAX_CYCLE_DURATION
        self.mean_score_per_generation.append(total_score / NUM_FEEDERS)
        if self.feeder_list[0].age >= MAX_CYCLE_DURATION:
            self.best_score_per_generation.append(100 + self.feeder_list[0].food_level)
        else:
            self.best_score_per_generation.append(100 * self.feeder_list[0].age / MAX_CYCLE_DURATION)
        if DISPLAY_GRAPH:
            self.graph_stats_per_generations()

    def graph_stats_per_generations(self):
        """
        A rather long method for drawing the graph of the best and mean scores per generation.
        """
        if len(self.best_score_per_generation) < 2:
            return
        graph_canvas = np.ones((GRAPH_SIZE,GRAPH_SIZE,3), dtype=float)
        cv2.line(img=graph_canvas, pt1=(GRAPH_MARGIN, GRAPH_MARGIN), pt2=(GRAPH_MARGIN, GRAPH_SIZE-GRAPH_MARGIN), color=(0, 0, 0), thickness=1)
        cv2.line(img=graph_canvas, pt1=(GRAPH_MARGIN, GRAPH_SIZE-GRAPH_MARGIN), pt2=(GRAPH_SIZE-GRAPH_MARGIN, GRAPH_SIZE-GRAPH_MARGIN), color=(0, 0, 0), thickness=1)
        best_score_ever = 0
        for score in self.best_score_per_generation:
            if score > best_score_ever:
                best_score_ever = score
        cv2.putText(img=graph_canvas, text="score", org=(5, 10), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(0,0,0))
        cv2.putText(img=graph_canvas, text=f"{best_score_ever:3.1f}", org=(GRAPH_MARGIN+5, GRAPH_MARGIN+8), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(0,0,0))
        cv2.putText(img=graph_canvas, text=f"gen: {len(self.best_score_per_generation)-1}", org=(GRAPH_SIZE-GRAPH_MARGIN-65, GRAPH_SIZE-GRAPH_MARGIN-14),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(0, 0, 0))

        vertical_scale = (GRAPH_SIZE-GRAPH_MARGIN-GRAPH_MARGIN)/best_score_ever
        horizontal_scale = (GRAPH_SIZE-GRAPH_MARGIN-GRAPH_MARGIN)/(len(self.best_score_per_generation)-1)

        horizontal_line_spacing = 10**(math.floor(math.log10(best_score_ever)))
        j = 1
        while j*horizontal_line_spacing < best_score_ever:
            cv2.line(img=graph_canvas,
                     pt1=(GRAPH_MARGIN, int(GRAPH_SIZE - GRAPH_MARGIN - j * horizontal_line_spacing * vertical_scale)),
                     pt2=(GRAPH_SIZE - GRAPH_MARGIN, int(GRAPH_SIZE - GRAPH_MARGIN - j * horizontal_line_spacing * vertical_scale)),
                     color=(0.75, 0.75, 0.75),
                     thickness=1
                     )
            j += 1

        vertical_line_spacing = 10**(math.floor(math.log10(len(self.best_score_per_generation)-1)))
        j = 1
        while j * vertical_line_spacing < len(self.best_score_per_generation):
            cv2.line(img=graph_canvas,
                     pt1=(int(GRAPH_MARGIN + j*vertical_line_spacing * horizontal_scale), GRAPH_MARGIN),
                     pt2=(int(GRAPH_MARGIN + j*vertical_line_spacing * horizontal_scale), GRAPH_SIZE-GRAPH_MARGIN),
                     color=(0.75, 0.75, 0.75),
                     thickness=1
                     )
            j += 1

        for i in range(len(self.best_score_per_generation)-1):
            cv2.line(img=graph_canvas,
                     pt1=(int(GRAPH_MARGIN + horizontal_scale*i), int(GRAPH_SIZE-GRAPH_MARGIN-vertical_scale*self.best_score_per_generation[i])),
                     pt2=(int(GRAPH_MARGIN + horizontal_scale*(i+1)), int(GRAPH_SIZE-GRAPH_MARGIN-vertical_scale*self.best_score_per_generation[i+1])),
                     color=(1, 0, 0), thickness=1)

            if horizontal_scale > 6:
                cv2.circle(img=graph_canvas, center= (int(GRAPH_MARGIN + horizontal_scale*i), int(GRAPH_SIZE-GRAPH_MARGIN-vertical_scale*self.best_score_per_generation[i])),
                           radius = 3, color=(1, 0, 0), thickness=-1)
            cv2.line(img=graph_canvas,
                     pt1=(int(GRAPH_MARGIN + horizontal_scale * i),
                          int(GRAPH_SIZE - GRAPH_MARGIN - vertical_scale * self.mean_score_per_generation[i])),
                     pt2=(int(GRAPH_MARGIN + horizontal_scale * (i + 1)),
                          int(GRAPH_SIZE - GRAPH_MARGIN - vertical_scale * self.mean_score_per_generation[i + 1])),
                     color=(0, 0, 1), thickness=1)
            if horizontal_scale > 6:
                cv2.circle(img=graph_canvas, center= (int(GRAPH_MARGIN + horizontal_scale*i), int(GRAPH_SIZE-GRAPH_MARGIN-vertical_scale*self.mean_score_per_generation[i])),
                           radius = 3, color=(0, 0, 1), thickness=-1)

        if horizontal_scale > 6:
            cv2.circle(img=graph_canvas,
                       center=(int(GRAPH_MARGIN + horizontal_scale * (i + 1)), int(GRAPH_SIZE - GRAPH_MARGIN - vertical_scale * self.best_score_per_generation[-1])),
                       radius=3, color=(1, 0, 0), thickness=-1)
            cv2.circle(img=graph_canvas,
                       center=(int(GRAPH_MARGIN + horizontal_scale * (i + 1)), int(
                           GRAPH_SIZE- GRAPH_MARGIN - vertical_scale * self.mean_score_per_generation[-1])),
                       radius=3, color=(0, 0, 1), thickness=-1)
        cv2.imshow("Graph",graph_canvas)

    def kill_all_feeders(self):
        """
        Time has expired for this generation, so  kill all the feeders (but preserve how much food each had.)
        """
        for bug in self.feeder_list:
            bug.die()

    def draw_all_feeders(self, main_canvas):
        """
        tell each live feeder to draw itself.
        :param main_canvas:
        """
        for bug in self.feeder_list:
            if bug.is_alive:
                bug.draw_self(canvas=main_canvas, display_sensors=DISPLAY_SENSORS)

    def count_live_feeders(self):
        """
        count how many feeders are still alive. If this number has dropped to zero, set self.cycle_ongoing to False.
        """
        self.live_feeders = 0
        for bug in self.feeder_list:
            if bug.is_alive:
                self.live_feeders += 1
        if self.live_feeders == 0:
            self.cycle_ongoing = False

    def update_stats_window(self):
        """
        draw the graphical representation of the feeders' genes, along with their age and cause of death, if any.
        Display the window.
        """
        if self.cycle_ongoing:
            self.feeder_list.sort(reverse=True)
            stats_canvas = np.ones((750, 600, 3), dtype=float)
            self.display_feeders(stats_canvas)
            cv2.imshow("stats", stats_canvas)

    def draw_all_food(self, main_canvas):
        """
        draw all the food dots on the canvas.
        :param main_canvas:
        """
        for f in self.food_list:
            f.draw_self(canvas=main_canvas)

    def check_for_feeder_danger_collisions(self):
        """
        determines whether any living feeders have collided with a danger, moving or non-moving. If so, the feeder should
        die.
        """
        for db in self.all_dangers:
            for bug in self.feeder_list:
                if bug.is_alive:
                    if math.pow(db.pos[0] - bug.position[0], 2) + math.pow(db.pos[1] - bug.position[1],
                                                                           2) < DANGER_THRESHOLD_SQUARED:
                        bug.die()
                        bug.food_level = 0
                        bug.death_reason = "O"

    def check_for_eaten_food(self):
        """
        for each food item, checks whether any feeder(s) is/are touching it. If so, increase the food_level of the
        feeder(s). Respawn the food at a new, random location.
        """
        eaten_food_list: Set[Food] = set()
        for f in self.food_list:
            for bug in self.feeder_list:
                if bug.is_alive:
                    if math.pow(f.pos[0] - bug.position[0], 2) + math.pow(f.pos[1] - bug.position[1],
                                                                          2) < FOOD_THRESHOLD_SQUARED:
                        bug.food_level = min(100, bug.food_level + 10)
                        eaten_food_list.add(f)
        for ef in eaten_food_list:
            self.food_list.remove(ef)
            self.food_list.append(Food())

    def move_all_feeders(self, delta_t):
        """
        perform one animation step for each live feeder.
        :param delta_t: the number of seconds since the last animation step.
        """
        for bug in self.feeder_list:
            if bug.is_alive:
                bug.animation_step(delta_t)

    def detect_all_food(self):
        """
        tell each feeder to update its sensors about all food in its range.
        """
        for f in self.food_list:
            for bug in self.feeder_list:
                if bug.is_alive:
                    bug.detect(f.pos, False)

    def detect_all_dangers(self):
        """
        tell each feeder to update its sensors about all dangers in its range.
        """
        for db in self.all_dangers:
            for bug in self.feeder_list:
                if bug.is_alive:
                    bug.detect(db.pos, True)

    def move_and_draw_dangers(self, delta_t, main_canvas):
        """
        animation step for all mobile dangers
        :param delta_t: the number of seconds since the last animation step
        :param main_canvas: the screen on which to draw them.
        """
        for db in self.moving_danger_list:
            db.animate_step(delta_t)
            if GRAPHIC_SIMULATION:
                db.draw_self(main_canvas)

    def clear_all_live_feeder_sensors(self):
        """
        refresh all the sensors for all the live feeders, in preparation to receive information about the world for
        this animation step.
        """
        for bug in self.feeder_list:
            if bug.is_alive:
                bug.clear_sensors()

    def initial_setup(self):
        """
        ask the user whether to load a data file of genes for a given generation.
        """
        load_YN = input("Do you want to load an existing generation? (Y/N) ").lower()
        if load_YN == 'y':
            load_filename = input("Enter the name of the file, or type 'cancel' to change your mind. ")
            if load_filename != "cancel":
                self.load_generation(filename=load_filename)

        self.save_filename = f"generation {self.program_run_number}"
        print("Click in the graphics window. "
              "Press 's' to save the current generation at the end of a cycle. "
              "Press 'q' to quit.")

    def save_generation(self, filename):
        """
        save information about the generation that just finished to a file, so that it can be loaded later.
        :param filename:
        """
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
        """
        read generation data from a file and set up the collection of feeders from this information, along with
        the "run number" and generation number to match the file.
        :param filename:
        """
        all_weights: List[List[float]] = []
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
        """
        Precondition: All the feeders have died off, either from collisions with dangers, starvation, or the simulation
        has expired. Self.feeder list is sorted by the success of the feeders, with the most successful at the start of
        the list. Note: some of the feeders at the bottom of the list might have been unlucky and spawned on top of a
        danger; you may wish to check the feeder's age to see whether it is below some threshold and give them a second
        chance. Or not... luck may be part of your breeding program!

        Postcondition: self.feeder_list contains NUM_FEEDERS feeders, new ones and/or rejuvenated returning ones, ready
        to act as the next generation. These might be:
        • returning successful feeders, rejuvenated
        • children of feeder pairs
        • mutated versions of feeders
        • random additions
        • other (e.g., combination children/mutated?)

        :return: None
        """
        # TODO: write this method, replacing the following code.
        # Dummy behavior. Just rejuvenates every Feeder, so the next generation is the same as this one.
        for bug in self.feeder_list:
            bug.rejuvenate()

if __name__ == "__main__":
    gar = GeneticAlgorithmRunner()
    gar.initial_setup()
    gar.animation_loop()
    cv2.destroyAllWindows()
    print("gen\tbest\tmean")
    for i in range(len(gar.best_score_per_generation)):
        print(f"{i}\t{gar.best_score_per_generation[i]:3.2f}\t{gar.mean_score_per_generation[i]:3.2f}")