from mission import DRLMission, FirstMission
from helpers.sim import State, make_simulation_object
from config import state_config
from pathlib import Path


CONE_HIT_PENALTY = 2


def time_map(mission: callable, map_path: str | Path) -> (bool, float):
    mission = mission()
    state = State(map_path, state_config)
    sim_runtime = make_simulation_object(state, mission)

    for i, (mission_time, observations, mission_action) in enumerate(sim_runtime):
        finished, steering_angle, speed_setpoint, _ = mission_action

        if finished:
            lap_time = mission_action[3]["lap_times"][0]
            cones_hit = state.cones_hit.sum()
            return True, lap_time, cones_hit

        elif not state.is_within_roi(120.):
            return False, 0., 0.


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
        total_time_w_cone_penalty = sum([result[1] + CONE_HIT_PENALTY * result[2] for result in results if result[0]])
        cones_hit = [result[2] for result in results if result[0]]

        print(f'\tAverage time (without cone penalty): {total_time / num_of_successes:.2f} seconds')
        print(f'\tAverage time (with cone penalty): {total_time_w_cone_penalty / num_of_successes:.2f} seconds')
        print(f'\tAverage number of cones hit: {sum(cones_hit) / num_of_successes:.2f}')


if __name__ == "__main__":
    # mission = FirstMission
    # time(mission)
    mission = DRLMission
    time(mission)
