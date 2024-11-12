from typing import Any
from helpers.path_tracking import stanley_steering
from helpers.path_planning import PathPlanner
from helpers.finish_detector import LapCounter
from helpers.speed_profile import SpeedProfile


class MyMission():
    """
    This is the class that defines the *mission logic* in our driverless system. 
    Files like these are used on the actual car. This mission represents the most barebones code that can 
    successfully run the autocross track.
    """

    def __init__(self):
        # Feel free to change these parameters
        self.path_planner = PathPlanner({"n_steps": 20, "verbose": False})
        self.lap_counter = LapCounter(6, 2., 10., [-0.5, 10, -4, 4]) 
        self.speed_profile = SpeedProfile(0.8, 2, 4)
        self.min_speed_setpoint = 2.5  # m/s # original 5
        self.max_safe_speed = 15 # float('inf')  # m/s # original 8
        self.speed_setpoint = 20 #self.min_speed_setpoint
        self.finished = False
        self.finish_time = float('inf')
        self.stopped_time = float('inf')

    def loop(self, args: dict, mission_time: float) -> tuple[bool, float, Any, dict[str, Any]]:
        percep_data = args["percep_data"]
        wheel_speed = args["actual_speed"]
        # 1. Path planning and speed profile
        path = self.path_planner.find_path(percep_data)
        try:
            self.speed_setpoint = self.speed_profile.profile(path, wheel_speed)[0]
        except Exception as e:
            print(f"Error in speed profile: {e}")

        self.speed_setpoint = 1.2 * max(min(self.speed_setpoint, self.max_safe_speed), self.min_speed_setpoint)
        #print(self.speed_setpoint)
        # 2. Stopping/finish logic
        self.lap_counter.update(percep_data.copy(), wheel_speed, mission_time)
        if self.lap_counter.lap_count >= 1:
            self.speed_setpoint = 0.0
            '''if wheel_speed <= 0.1:
                self.stopped_time = min(mission_time, self.stopped_time)
                print (self.stopped_time)
        if self.stopped_time + 1. < mission_time:'''
            self.finished = True
        # 3. controls, you SHOULD tune the constants here
        steering_ang, controller_log = stanley_steering(path, 3.5, wheel_speed)
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
