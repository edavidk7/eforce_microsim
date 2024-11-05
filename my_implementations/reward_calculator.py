from helpers.sim import State
import numpy as np


DISTANCE_THRESHOLD = 10
MIN_REWARD = -10
MAX_REWARD = 8


class RewardScaler:
    @staticmethod
    def scale_reward(reward: float) -> float:
        return (reward - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)


class RewardCalculator:
    def __init__(self) -> None:
        self.centerline = None
        self.cones_hit = 0

    def _compute_centerline(self, state: State) -> None:
        """
        Compute the centerline of the track from the cone positions
        """
        num_points = min(len(state.yellow_cones), len(state.blue_cones))
        blue_cones = state.blue_cones[:num_points, :2]
        yellow_cones = state.yellow_cones[:num_points, :2]

        self.centerline = (blue_cones + yellow_cones) / 2

    def _get_nearest_centerline_point_distance(self, state: State) -> float:
        """
        Compute the distance between the car and the nearest point on the centerline
        """
        car_pos = state.car_pose[:2]
        distances = np.linalg.norm(self.centerline - car_pos, axis=1)
        min_index = np.argmin(distances)
        return distances[min_index]

    def compute_reward(self, state: State) -> float:
        """
        Compute the reward based on the current state
        """

        # Compute the centerline only once (needed for reward computation)
        if self.centerline is None:
            self._compute_centerline(state)

        reward = 0.0

        # Penalize old cone hits (low penalty)
        reward -= 1.0 * np.sum(state.cones_hit)

        # Penalize new cone hits (high penalty)
        new_hits = np.sum(state.cones_hit) - self.cones_hit
        self.cones_hit += new_hits
        reward -= 20 * new_hits

        # Small penalties for abrupt changes in steering angle and excessive yaw rate
        reward -= 0.1 * (abs(state.steering_angle) ** 2)
        reward -= 0.1 * abs(state.velocity[2])

        # Penalize deviation from the centerline
        deviation = self._get_nearest_centerline_point_distance(state)
        if deviation > DISTANCE_THRESHOLD:
            reward -= 15 * (deviation - DISTANCE_THRESHOLD)
        else:
            # Speed encouragement
            # (only if the car is not off track)
            reward += 0.01 * state.noisy_speed

        return RewardScaler.scale_reward(reward)
