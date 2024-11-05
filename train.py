from mission import DRLMission
from my_implementations.speed_profile import SpeedProfiler
from helpers.sim import State, make_simulation_object
from config import state_config
from pathlib import Path
import time


def run_map(model: SpeedProfiler, map_path: str | Path) -> None:
    mission = DRLMission(model)
    state = State(map_path, state_config)
    sim_runtime = make_simulation_object(state, mission, provide_state_information=True)

    for i, (mission_time, observations, mission_action) in enumerate(sim_runtime):
        finished, steering_angle, speed_setpoint, _ = mission_action

        if finished:
            print(f"Finished in {mission_time:.2f} seconds!")
            break

        elif not state.is_within_roi(120.):
            print("Car went out of bounds!")
            break


def run_one_simulation(model: SpeedProfiler) -> None:
    for i in range(1, 4):
        map_path = Path(f"maps/map{i}.json")
        run_map(model, map_path)


def run_for_n_iter(n: int, model: SpeedProfiler) -> None:
    for _ in range(n):
        run_one_simulation(model)


def run_for_n_seconds(n: int, model: SpeedProfiler) -> None:
    start = time.time()
    while time.time() - start < n:
        run_one_simulation(model)


if __name__ == "__main__":
    model = SpeedProfiler(train=True, logging=True)
    run_for_n_iter(100, model)
    model.save_model()
