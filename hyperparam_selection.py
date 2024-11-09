#!/usr/bin/env python3

from test import time_map, CONE_HIT_PENALTY
from config import state_config
from pathlib import Path
import random
import multiprocessing
from functools import total_ordering
from mission import EVAMission
from my_implementations.speed_profile import SpeedProfiler


RANGES = [
    (0, 10),  # min speed
    (10, 30),  # max speed
    (1, 10),  # lookahead distance
    (0, 10),  # gain
    (0, 10),  # lateran gain
    (0, 10),  # kp
    (0, 10),  # ki
    (0, 10),  # kd
]
POP_SIZE = 10
CR = 0.9
FAIL_PENALTY = 50
TIMEOUT_LIMIT = 100
NUM_SIMS = 3
speed_profiler = SpeedProfiler()


@total_ordering
class Individual:
    def __init__(self, hyperparameters: list) -> None:
        self.hyperparameters = hyperparameters
        self.fitness = None
        self.age = 0

    def evaluate(self) -> float:
        if self.fitness is None:
            # Run the simulation
            results = []
            state_config['controller_gains'] = {
                'kp': self.hyperparameters[5],
                'ki': self.hyperparameters[6],
                'kd': self.hyperparameters[7],
            }

            def simulate(queue: multiprocessing.Queue) -> None:
                results = []
                for i in range(1, 4):
                    map_path = Path(f"maps/map{i}.json")
                    mission = EVAMission(self.hyperparameters, speed_profiler)

                    for _ in range(NUM_SIMS):
                        results.append(time_map(mission, map_path, state_config, init_mission=False))
                queue.put(results)

            # Run the simulation in a separate process with timeout
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=simulate, args=(queue,))
            process.start()
            process.join(TIMEOUT_LIMIT)

            # Punish the individual if it times out
            if process.is_alive():
                process.terminate()
                process.join()
                self.fitness = FAIL_PENALTY * 3 * NUM_SIMS

            else:
                results = queue.get()
                # Calculate fitness
                self.fitness = 0
                for success, time, cones in results:
                    if not success:
                        self.fitness += FAIL_PENALTY
                    else:
                        self.fitness += time + CONE_HIT_PENALTY * cones
                print(len(results), self.fitness)

        return self.fitness

    def __eq__(self, other) -> bool:
        return self.evaluate() == other.evaluate()

    def __lt__(self, other) -> bool:
        return self.evaluate() < other.evaluate()

    def __str__(self) -> str:
        return f'Individual: {self.hyperparameters}, Fitness: {self.fitness}'


class DifferentialEvolution:
    def __init__(self) -> None:
        self.population = [Individual([random.uniform(*interval) for interval in RANGES]) for _ in range(POP_SIZE)]

    @staticmethod
    def crossover(original_params: list, parent: Individual) -> Individual:
        child_params = []
        for i in range(len(RANGES)):
            if random.random() < CR:
                child_params.append(max(RANGES[i][0], min(original_params[i], RANGES[i][1])))
            else:
                child_params.append(parent.hyperparameters[i])

        return Individual(child_params)

    def create_offspring(self, parent: Individual) -> Individual:
        a, b, c = random.sample(self.population, 3)
        F = random.uniform(0.5, 1.0)

        offspring_params = [a.hyperparameters[i] + F * (b.hyperparameters[i] - c.hyperparameters[i]) for i in range(len(RANGES))]
        offspring = DifferentialEvolution.crossover(offspring_params, parent)

        if offspring <= parent:
            return offspring
        else:
            return parent

    def get_best(self) -> Individual:
        return min(self.population)

    def evolve(self) -> None:
        new_population = []

        for parent in self.population:
            new_population.append(self.create_offspring(parent))

        self.population = new_population
        for individual in self.population:
            individual.age += 1

    def log(self, gen: int) -> None:
        print(f'Generation {gen}:')
        print(f'\t{self.get_best()}')

    def run(self, n: int) -> None:
        for gen in range(n):
            self.evolve()
            self.log(gen)


if __name__ == "__main__":
    de = DifferentialEvolution()
    de.run(100)
    print(de.get_best())
