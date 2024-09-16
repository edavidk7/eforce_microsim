import numpy as np


class SpeedProfile:
    def __init__(self, mu=0.8, max_acceleration=2, max_braking=4):
        self.GRAVITY = 9.8
        self.CAR_MU = mu
        self.MAX_ACCELERATION = max_acceleration
        self.MAX_BRAKING = max_braking

    def lengths(self, path):
        # Compute segment lengths between consecutive path points
        segment_length = np.zeros(len(path) - 1)
        for i in range(1, len(path)):
            segment_length[i - 1] = np.linalg.norm(path[i] - path[i - 1])
        return segment_length

    def curvatures(self, path):
        def curvature(origin, current, destination):
            # Calculate curvature from three consecutive points
            a = np.linalg.norm(current - origin)
            b = np.linalg.norm(destination - current)
            c = np.linalg.norm(destination - origin) + 1e-6

            q = (a**2 + b**2 - c**2) / (2 * a * b + 1e-6)
            return (2 * np.sqrt(max(0., 1 - q**2))) / c

        # Compute curvatures for each path segment
        segment_curvature = np.zeros(len(path))
        for i in range(1, len(path) - 1):
            segment_curvature[i] = curvature(path[i - 1], path[i], path[i + 1])
        segment_curvature[0] = 0.
        segment_curvature[-1] = 0.
        return segment_curvature + 0.001  # A small offset to prevent division by zero

    def profile(self, path, initial_speed=0.):
        def stanford_profile(seglengths, segcurvatures):
            # First pass: Speed based on curvature and friction ellipse
            first_pass = np.sqrt((self.CAR_MU * self.GRAVITY) / segcurvatures)

            # Second pass: Limit speed to avoid extreme acceleration
            second_pass = np.zeros_like(first_pass)
            second_pass[0] = initial_speed
            for s in range(1, len(first_pass)):
                compensation = 2 * self.MAX_ACCELERATION * seglengths[s - 1]
                second_pass[s] = min(first_pass[s], np.sqrt(second_pass[s - 1]**2 + compensation))

            # Third pass: Limit speed to avoid extreme braking
            third_pass = np.copy(second_pass)
            for s in range(len(second_pass) - 2, -1, -1):
                compensation = 2 * self.MAX_BRAKING * seglengths[s]
                third_pass[s] = min(second_pass[s], np.sqrt(third_pass[s + 1]**2 + compensation))

            return third_pass

        # Compute curvatures and segment lengths
        curvatures = self.curvatures(path)
        seglengths = self.lengths(path)

        # Compute the speed profile
        speed_arr = stanford_profile(seglengths, curvatures)
        target_speed = speed_arr[1] if len(speed_arr) > 1 else speed_arr[0]

        # Log information
        speedprofile_log = {
            "curvatures": curvatures,
            "speed_arr": speed_arr,
            "target_speed": target_speed,
        }
        return target_speed, speedprofile_log
