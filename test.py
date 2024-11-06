from mission import DRLMission, FirstMission
from helpers.sim import State, make_simulation_object
from config import state_config
from pathlib import Path


def time_map(mission: callable, map_path: str | Path) -> (bool, float):
    mission = mission()
    state = State(map_path, state_config)
    sim_runtime = make_simulation_object(state, mission)

    for i, (mission_time, observations, mission_action) in enumerate(sim_runtime):
        finished, steering_angle, speed_setpoint, _ = mission_action

        if finished:
            return True, mission_time

        elif not state.is_within_roi(120.):
            return False, 0.


def time(mission: callable, num_sims: int = 10) -> None:
    for i in range(1, 4):
        map_path = Path(f"maps/map{i}.json")
        results = []

        for _ in range(num_sims):
            results.append(time_map(mission, map_path))

        print(f'Map {i}:')

        num_of_successes = sum([result[0] for result in results])
        print(f'\tSuccess rate: {num_of_successes / num_sims * 100:.2f}% ({num_of_successes} / {num_sims})')

        total_time = sum([result[1] for result in results if result[0]])
        print(f'\tAverage time: {total_time / num_of_successes:.2f} seconds')


if __name__ == "__main__":
    mission = FirstMission
    time(mission)
    mission = DRLMission
    time(mission)
