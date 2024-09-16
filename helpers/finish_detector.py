import numpy as np
from .cone_utils import filter_occluded_cones
"""
Finish detecter keeps count of recent big cone detections
if this count is high enough, finish is detected
it is detected as a timestamp, computed by finish distance and car's speed
"""


class FinishDetector():
    def __init__(self, min_detections, detection_timeout, occlusion_profile):
        # detections of big orange cones and their timestamps
        self.big_orange_detections = np.zeros((0, 3))  # timestamp, x, y
        self.min_detections = min_detections  # number of detections to be considered as finish line
        self.detection_timeout = detection_timeout  # time in seconds after which the detections are cleared
        self.occlusion_profile = occlusion_profile

    def update(self, cone_preds, current_time):
        """
        given cone predictions, returns distance of the car to mean position of big orange cones
        args:
          cone_preds - Nx3 np.array of cone positions [x,y,cls]
          min_big_cones - minimum count of big orange cones for the distance to be calculated
        rets:
          dist_to_finish
        """
        cone_preds = filter_occluded_cones(cone_preds, self.occlusion_profile)
        big_cone_idxs = np.where(cone_preds[:, 2] == 3)[0]
        big_cones = cone_preds[big_cone_idxs, 0:2]

        # filter detections older than detection_timeout
        self.big_orange_detections = self.big_orange_detections[self.big_orange_detections[:, 0] > current_time - self.detection_timeout]

        # if no big orange cones are being detected, return None
        if big_cones.shape[0] == 0:
            return None

        # expand big cone dim with current timestamp & append to big_orange_detections
        big_cones_with_timestamp = np.hstack((np.ones((big_cones.shape[0], 1)) * current_time, big_cones))
        self.big_orange_detections = np.vstack((self.big_orange_detections, big_cones_with_timestamp))

        if self.big_orange_detections.shape[0] >= self.min_detections:
            distance_to_finish = np.linalg.norm(np.average(big_cones, axis=0))
            if distance_to_finish < 4:
                return distance_to_finish
            else:
                return None

        return None


class LapCounter():
    def __init__(self, min_detections, detection_timeout, min_lap_time, occlusion_profile):
        self.min_lap_time = min_lap_time
        self.finish_detector = FinishDetector(min_detections, detection_timeout, occlusion_profile)
        self.lap_count = 0
        self.next_lap_stamp = float("inf")
        self.prev_lap_stamp = 0.0
        self.lap_times = []

    def update(self, cone_preds, speed, current_time):
        if current_time > self.next_lap_stamp:
            self.lap_count += 1
            self.lap_times.append(current_time - self.prev_lap_stamp)
            self.prev_lap_stamp = current_time
            self.next_lap_stamp = float("inf")
        self.lap_time = current_time - self.prev_lap_stamp
        if current_time > self.prev_lap_stamp + self.min_lap_time:
            distance_to_finish = self.finish_detector.update(cone_preds, current_time)
            if distance_to_finish is not None:
                time_to_finish = distance_to_finish / (speed + 1e-6)
                self.next_lap_stamp = current_time + time_to_finish
