from typing import Any
from helpers.path_tracking import stanley_steering
from helpers.path_planning import PathPlanner
from helpers.finish_detector import LapCounter
from helpers.speed_profile import SpeedProfile
import math
import numpy as np

class MyMission():
    
    current_map_index = 0
    map_types = ["map1", "map2", "map3"]

    def __init__(self):
        
        self.map_type = MyMission.map_types[MyMission.current_map_index]
        self.set_threshold()   

        # Inicializace 
        self.path_planner = PathPlanner({"n_steps": 20, "verbose": False})
        self.lap_counter = LapCounter(6, 2., 10., [-0.5, 10, -4, 4])
        self.speed_profile = SpeedProfile(0.8, 2, 4)
        self.speed_min = 5.0
        self.max_speed_setpoint = 15.0
        self.max_safe_speed = 30.0
        self.max_speed_map3 = 13.0
        self.min_speed_setpoint = self.speed_min
        self.finished = False
        self.finish_time = float('inf')
        self.stopped_time = float('inf')

    def set_threshold(self):
        # threshold 
        if self.map_type == "map1":
            self.threshold_base = lambda angle_diff: 4.5 * np.exp(-3.1 * angle_diff) + 12.0
        elif self.map_type == "map2":
            self.threshold_base = lambda angle_diff: 4.0 * np.exp(-4.1 * angle_diff) + 9.0
        elif self.map_type == "map3":
            self.threshold_base = lambda angle_diff: 3.5 * np.exp(-4.1 * angle_diff) + 8.0

    def categorize_turn(self, path, current_heading, look_ahead_steps=1):
        
        if len(path) <= look_ahead_steps:
            return self.max_speed_setpoint, 0.0  
        
        # Calculate angle difference between current and look-ahead points
        look_ahead_point = path[look_ahead_steps]
        dx = look_ahead_point[0] - path[0][0]
        dy = look_ahead_point[1] - path[0][1]
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(current_heading - target_angle)

        
        threshold = self.threshold_base(angle_diff)
        
        
        return threshold, angle_diff

    def loop(self, args: dict, mission_time: float) -> tuple[bool, float, Any, dict[str, Any]]:
        percep_data = args["percep_data"]
        wheel_speed = args["actual_speed"]
        current_heading = args.get("heading", 0.0)  
       
        # 1. Path planning and categorization of turn
        path = self.path_planner.find_path(percep_data)
        self.speed_setpoint, angle_diff = self.categorize_turn(path, current_heading)
        gain = 5

         
        if self.map_type == "map3":
            self.speed_setpoint = min(self.speed_setpoint, self.max_speed_map3)

        # 2. Stopping/finish logic
        self.lap_counter.update(percep_data.copy(), wheel_speed, mission_time)

        #next map
        if self.lap_counter.lap_count >= 1 and wheel_speed <= 0.1:
            if MyMission.current_map_index < len(MyMission.map_types) - 1:
                
                MyMission.current_map_index += 1
                self.finished = True  
                print(f"Switching to {MyMission.map_types[MyMission.current_map_index]}")  
            else:
                self.finished = True  

        
        if self.lap_counter.lap_count >= 1:
            self.speed_setpoint = 0.0
            if wheel_speed <= 0.1:
                self.stopped_time = min(mission_time, self.stopped_time)
        if self.stopped_time + 1. < mission_time:
            self.finished = True

        # 3. stanley_steering controls
        steering_ang, controller_log = stanley_steering(path, gain, wheel_speed, 2.5, 0)

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

