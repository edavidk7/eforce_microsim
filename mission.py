from typing import Any
from bros.algorithms.path_tracking import stanley_steering
from bros.algorithms.path_planning import PathPlanner
from bros.algorithms.finish_detector import LapCounter


class MyMission():
    """
    This is the class that defines the *mission logic* in our driverless system. 
    Files like these are used on the actual car. This mission represents the most barebones code that can 
    successfully run the autocross track.
    """

    def __init__(self):
        self.path_planner = PathPlanner({"n_steps": 20, "verbose": False})
        self.lap_counter = LapCounter(6, 2., 10., [-0.5, 10, -4, 4])
        self.speed_setpoint = 8.  # m/s
        self.finished = False
        self.finish_time = float('inf')
        self.stopped_time = float('inf')

    def loop(self, args: dict, mission_time: float) -> tuple[bool, float, Any, dict[str, Any]]:
        percep_data = args["percep_data"]
        wheel_speed = args["actual_speed"]
        self.lap_counter.update(percep_data.copy(), wheel_speed, mission_time)
        # 1. Stopping/finish logic
        if self.lap_counter.lap_count >= 1:
            self.speed_setpoint = 0.0
            if wheel_speed <= 0.1:
                self.stopped_time = min(mission_time, self.stopped_time)
        if self.stopped_time + 1. < mission_time:
            self.finished = True
        # 2. path planning
        path = self.path_planner.find_path(percep_data)
        # 3. controls
        steering_ang, controller_log = stanley_steering(path, 4.5, wheel_speed, 2.9, 0.0)
        # 4. logging and debugging
        extras = {
            "mission_time": mission_time,
            "finish_time": self.finish_time,
            "path": path,
            "lap_times": self.lap_counter.lap_times,
            "controller_log": controller_log,
        }
        # 5. return
        return self.finished, self.speed_setpoint, steering_ang, extras
