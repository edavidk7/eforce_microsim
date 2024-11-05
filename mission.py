from typing import Any
from helpers.path_tracking import stanley_steering
from helpers.finish_detector import LapCounter
from helpers.speed_profile import SpeedProfile
from helpers.sim import State
from my_implementations.path_planning import PathPlanner
from my_implementations.speed_profile import SpeedProfiler as DRLSpeedProfiler
from my_implementations.reward_calculator import RewardCalculator


class DRLMission:
    """
    Mission class using my path planner (Delaunay) and speed profiler (DRL model)
    """

    def __init__(self, model: DRLSpeedProfiler | None = None, logging: bool = False) -> None:
        # my implementations
        self.path_planner = PathPlanner()
        self.speed_profile = model or DRLSpeedProfiler(train=False, logging=logging)
        self.reward_calculator = RewardCalculator()

        self.lap_counter = LapCounter(6, 2., 10., [-0.5, 10, -4, 4])
        self.min_speed_setpoint = 5.  # m/s
        self.max_safe_speed = 16.  # m/s
        self.speed_setpoint = self.min_speed_setpoint
        self.finished = False
        self.finish_time = float('inf')
        self.stopped_time = float('inf')

    def loop(self, args: dict, mission_time: float, state: State | None = None) -> tuple[bool, float, Any, dict[str, Any]]:
        percep_data = args["percep_data"]
        wheel_speed = args["actual_speed"]
        steering_angle = args["actual_steering_angle"]

        # 0. Get reward from the environment (only during training)
        if self.speed_profile.train and state is not None:
            reward = self.reward_calculator.compute_reward(state)
        else:
            reward = None

        # 1. Path planning and speed profile
        path = self.path_planner.find_path(percep_data)
        try:
            self.speed_setpoint += self.speed_profile.predict(path, wheel_speed, steering_angle, reward)
        except Exception as e:
            print(f"Error in speed profile: {e}")

        self.speed_setpoint = max(min(self.speed_setpoint, self.max_safe_speed), self.min_speed_setpoint)

        # 2. Stopping/finish logic
        self.lap_counter.update(percep_data.copy(), wheel_speed, mission_time)
        if self.lap_counter.lap_count >= 1:
            self.speed_setpoint = 0.0
            if wheel_speed <= 0.1:
                self.stopped_time = min(mission_time, self.stopped_time)
        if self.stopped_time + 1. < mission_time:
            self.finished = True

        # 3. controls, you SHOULD tune the constants here
        steering_ang, controller_log = stanley_steering(path, 3, wheel_speed, 1.8, 0.6)

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

    def save(self) -> None:
        """
        Save the model (should be called after training)
        """
        self.speed_profile.save_model()


class FirstMission:
    """
    Mission class using my path planner (Delaunay) and provided speed profiler
    """

    def __init__(self):
        # my implementations
        self.path_planner = PathPlanner()
        self.speed_profile = SpeedProfile(0.8, 2, 4)
        self.lap_counter = LapCounter(6, 2., 10., [-0.5, 10, -4, 4])
        self.min_speed_setpoint = 5.  # m/s
        self.max_safe_speed = 16.  # m/s
        self.speed_setpoint = self.min_speed_setpoint
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

        self.speed_setpoint = max(min(self.speed_setpoint, self.max_safe_speed), self.min_speed_setpoint)

        # 2. Stopping/finish logic
        self.lap_counter.update(percep_data.copy(), wheel_speed, mission_time)
        if self.lap_counter.lap_count >= 1:
            self.speed_setpoint = 0.0
            if wheel_speed <= 0.1:
                self.stopped_time = min(mission_time, self.stopped_time)
        if self.stopped_time + 1. < mission_time:
            self.finished = True

        # 3. controls, you SHOULD tune the constants here
        steering_ang, controller_log = stanley_steering(path, 3, wheel_speed, 1.8, 0.6)

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
