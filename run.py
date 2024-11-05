import datetime
from argparse import ArgumentParser
from mission import MyMission
from pathlib import Path
from helpers.sim import State, StateRenderer, make_simulation_object, plot_state_summary_and_wait, history_to_csv
from config import state_config

MAP_ROI = 120.  # meters


def run_mission(map_path: str | Path, logdir: Path | None, render: bool):
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
            finish_time = lap_time
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
    args = parser.parse_args()
    if args.no_logs:
        logdir = None
        print("Not logging")
    else:
        logdir = Path("logs") / datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        print(f"Logging to {logdir}")
    if args.no_render:
        render = False
    else:
        render = True
    for i in range(1, 4):
        MAP_PATH = Path(f"maps/map{i}.json")
        if logdir is not None:
            subdir = logdir / f"map{i}"
            subdir.mkdir(parents=True, exist_ok=True)
        else:
            subdir = None
        print(f"Running mission on map {i}")
        run_mission(MAP_PATH, subdir, render)
