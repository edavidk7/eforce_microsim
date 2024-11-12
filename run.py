'''import datetime
from argparse import ArgumentParser
from mission import MyMission
from pathlib import Path
from helpers.sim import State, StateRenderer, make_simulation_object, plot_state_summary_and_wait, history_to_csv
from config import state_config
from gains import controller_gains

MAP_ROI = 100.  # meters # original 120.


def run_mission(map_path: str | Path, logdir: Path | None):
    state = State(map_path, state_config)
    mission = MyMission()
    sim_runtime = make_simulation_object(state, mission)
    renderer = StateRenderer()
    renderer.set_state(state)
    finish_time = None
    for i, (mission_time, observations, mission_action) in enumerate(sim_runtime):
        speed_setpoint = mission_action[1]
        printout = f"Time: {mission_time: .2f}\nSpeed: {state.speed: .2f}\nSetpoint: {speed_setpoint: .2f}\n"
        renderer.render_state(printout, path=mission_action[3]["path"])
        finished, steering_angle, speed_setpoint, _ = mission_action
        if finished == True:
            #print(f"Finished in {mission_time:.2f} seconds!")
            finish_time = mission_time
            break
        elif not state.is_within_roi(MAP_ROI):
            #print("Car went out of bounds!")
            break
    renderer.close()
    if logdir is not None:
        history_to_csv(state, logdir / "history.csv")
    plot_state_summary_and_wait(state, finish_time, success=finish_time is not None)
    if finished:
        return mission_time, state.cones_world[state.cones_hit]
    elif not state.is_within_roi(MAP_ROI):
        return 'DNF', state.cones_world[state.cones_hit]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--no_logs", action="store_true", default=False)
    args = parser.parse_args()
    if args.no_logs:
        logdir = None
        print("Not logging")
    else:
        logdir = Path("logs") / datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        print(f"Logging to {logdir}")
    file_names = ['results_map1', 'results_map2', 'results_map3']
    for i in range(1,4): # maps driven
        MAP_PATH = Path(f"maps/map{i}.json")
        if logdir is not None:
            subdir = logdir / f"map{i}"
            subdir.mkdir(parents=True, exist_ok=True)
        else:
            subdir = None
        print(f"Running mission on map {i}")
        #print(run_mission(MAP_PATH, subdir))
        for rep in range(1):
            time, hit_cones = run_mission(MAP_PATH, subdir)
            print(str(time) + ' + 2 * ' + str(len(hit_cones)))
'''

import datetime
from argparse import ArgumentParser
from mission import MyMission
from pathlib import Path
from helpers.sim import State, StateRenderer, make_simulation_object, plot_state_summary_and_wait, history_to_csv
from config import state_config

MAP_ROI = 120.  # meters


def run_mission(map_path: str | Path, logdir: Path | None, render: bool, benchmark: bool = False):
    state = State(map_path, state_config)
    mission = MyMission()
    sim_runtime = make_simulation_object(state, mission)
    if render: renderer = StateRenderer()
    if render: renderer.set_state(state)
    finish_time = None
    for i, (mission_time, observations, mission_action) in enumerate(sim_runtime):
        speed_setpoint = mission_action[1]
        printout = f"Time: {mission_time: .2f}\nSpeed: {state.speed: .2f}\nSetpoint: {speed_setpoint: .2f}\n"
        if render: renderer.render_state(printout, path=mission_action[3]["path"])
        finished, steering_angle, speed_setpoint, _ = mission_action
        if finished == True:
            lap_time = mission_action[3]["lap_times"][0]
            print(f"Finished lap in {lap_time:.2f} seconds!")
            if benchmark == True:
                print(f"Number of cones hit: {state.cones_hit.sum()}")
                print(f"Benchmark time: {(lap_time + 2.*state.cones_hit.sum()):.2f}")
            finish_time = lap_time
            if benchmark: finish_time += 2.*state.cones_hit.sum()
            break
        elif not state.is_within_roi(MAP_ROI):
            print("Car went out of bounds!")
            break
    if render: renderer.close()
    if logdir is not None:
        history_to_csv(state, logdir / "history.csv")
    plot_state_summary_and_wait(state, finish_time, success=finish_time is not None)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--no_logs", action="store_true", default=False)
    parser.add_argument("--no_render", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    args = parser.parse_args()
    if args.no_logs:
        logdir = None
        print("Not logging")
    else:
        logdir = Path("logs") / datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        print(f"Logging to {logdir}")
    for i in range(1, 4):
        MAP_PATH = Path(f"maps/map{i}.json")
        if logdir is not None:
            subdir = logdir / f"map{i}"
            subdir.mkdir(parents=True, exist_ok=True)
        else:
            subdir = None
        print(f"Running mission on map {i}")
        run_mission(MAP_PATH, subdir, not args.no_render, args.benchmark)
