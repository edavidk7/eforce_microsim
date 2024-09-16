import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from scipy.stats import skewnorm
from config import ConeClasses


def filter_occluded_cones(cones_local, occlusion_profile):
    """
    Filters map_dict cones outside of the occlusion profile
    args:
      cones_local - Nx3 np.array
      occlusion_profile - 4 element tuple of floats (y_min, y_max, x_min, x_max) - see config.py
    ret:
      cones_filtered - Mx3 np.array
    """
    forward_min, forward_max, left_max, right_max = occlusion_profile
    mask_lr = (cones_local[:, 0] <= forward_max) & (cones_local[:, 0] >= forward_min)
    mask_f = (cones_local[:, 1] <= right_max) & (cones_local[:, 1] >= left_max)
    mask = mask_lr & mask_f
    cones_filtered = cones_local[mask, :]
    return cones_filtered


def get_occlusion_mask(cones_local, occlusion_profile):
    forward_min, forward_max, left_max, right_max = occlusion_profile
    mask_lr = (cones_local[:, 0] <= forward_max) & (cones_local[:, 0] >= forward_min)
    mask_f = (cones_local[:, 1] <= right_max) & (cones_local[:, 1] >= left_max)
    mask = mask_lr & mask_f
    return mask


def get_fov_mask(points, max_dist, fov):
    mask_dist = np.linalg.norm(points[:, 0:2], axis=1) < max_dist

    # Calculate y values for the FOV lines
    tan_fov_2 = np.tan(np.radians(fov / 2))
    y1 = tan_fov_2 * points[:, 0]
    y2 = -tan_fov_2 * points[:, 0]

    # Find points within the FOV cone
    mask_fov = (points[:, 1] < y1) & (points[:, 1] > y2)
    mask = mask_dist & mask_fov
    return mask


def filter_points_by_fov(points, max_dist, fov):
    """
    Filters points by distance and FOV
    Args:
        points: Nx2 np.array of points
        max_dist: float
        fov: float
    Returns:
        filtered_points: Nx2 np.array
    """
    mask_dist = np.linalg.norm(points[:, 0:2], axis=1) < max_dist

    # Calculate y values for the FOV lines
    tan_fov_2 = np.tan(np.radians(fov / 2))
    y1 = tan_fov_2 * points[:, 0]
    y2 = -tan_fov_2 * points[:, 0]

    # Find points within the FOV cone
    mask_fov = (points[:, 1] < y1) & (points[:, 1] > y2)
    mask = mask_dist & mask_fov
    filtered_points = points[mask, :]

    return filtered_points


def apply_noise(cones_local, stochastic_sim_config, occlusion_profile):
    # 1. apply noise to the position of the cones
    # A. use skew-normal distribution for the noise in the x-direction
    random_skew_normal_noise = skewnorm.rvs(
        a=stochastic_sim_config["detection_pos_shape_x"],  # shape parameter (α)
        loc=0,  # location parameter (ξ)
        scale=stochastic_sim_config["detection_pos_sigma_x"],  # scale parameter (ω)
        size=cones_local.shape[0]
    )
    cones_local[:, 0] += random_skew_normal_noise

    # B. use standard normal distribution in the y-direction
    cones_local[:, 1] += np.random.normal(0, stochastic_sim_config["detection_pos_sigma_y"], cones_local.shape[0])

    # 2. flip cone classes with a probability
    flip_matrix = stochastic_sim_config["cone_class_flip_matrix"]
    for i in range(cones_local.shape[0]):
        cones_local[i, 2] = np.random.choice([0, 1, 2, 3], p=flip_matrix[int(cones_local[i, 2])])

    # 3. remove cones with a probability
    mask = np.ones(cones_local.shape[0], dtype=bool)
    for i in range(cones_local.shape[0]):
        cones_cls = int(cones_local[i, 2])
        if np.random.rand() < stochastic_sim_config["false_negative_prob"][cones_cls]:
            mask[i] = False
    cones_local = cones_local[mask]

    # 4. create FP cones with a probability
    while True:
        if np.random.rand() < stochastic_sim_config["spurious_detection_prob"]:  # generate new cone with uniform distribution in range 1 to 5
            min_y, max_y, min_x, max_x = occlusion_profile
            new_cone = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y), np.random.choice([0, 1, 2, 3])])
            cones_local = np.vstack((cones_local, new_cone))
        else:
            break

    return cones_local


def get_big_orange_distance(cone_preds, min_big_cones):
    """
    Given cone predictions, returns distance of the car to mean position of big orange cones
    args:
      cone_preds - Nx3 np.array of cone positions [x,y,cls]
      min_big_cones - minimum count of big orange cones for the distance to be calculated
    rets:
      dist_to_finish
    """
    big_cone_idxs = np.where(cone_preds[:, 2] == 3)[0]
    if len(big_cone_idxs) >= min_big_cones:
        distance_to_finish = np.mean(cone_preds[big_cone_idxs, 0])
        return distance_to_finish
    else:
        return None


def get_orange_centerline(cone_preds):
    """
    Finds center points between 2 opposing cones, for finding the path in skidpad with only orange cones.
    Args:
        cone_preds (numpy.ndarray N x 3): cone positions [x,y,cls]
    Returns:
        numpy.ndarray [N/2 x 2]: center points between supplied cone positions
    """
    dists = np.sqrt(cone_preds[:, 0]**2 + cone_preds[:, 1]**2)
    cone_preds = cone_preds[np.argsort(dists, axis=0)]
    means = [[0., 0.]]
    for i in range(0, cone_preds.shape[0] - (cone_preds.shape[0] % 2 != 0), 2):
        means.append(np.mean(cone_preds[i:i + 2, 0:2], axis=0))
    return np.array(means)


def recolor_orange_cones(cone_preds):
    """Recolors orange cones by left and right, assuming the car is in the middle of the track
    Args:
        cone_preds (np.array): big orange cone predictions
    Returns:
        np.array: recolored orange cones by left and right
    """
    for cone in cone_preds:
        if cone[2] == ConeClasses.ORANGE or cone[2] == ConeClasses.BIG:
            if cone[1] > 0.0:
                # Make all cones on the left blue
                cone[2] = ConeClasses.BLUE.value
            elif cone[1] < 0.0:
                # Make all cones on the right yellow
                cone[2] = ConeClasses.YELLOW.value
    return cone_preds


def load_map_from_json(json_path: Path) -> Dict[str, Union[np.ndarray, float]]:
    """Loads map from json file

    Args:
        json_path (Path): path to json file

    Returns:
        Dict[str, Union[np.ndarray, float]]: dictionary with map data, the following keys are present:

        car_position: [x, y] - car start position (np.ndarray)

        car_heading: float - car start heading (degrees)

        start_line: [[x1, y1], [x2, y2]] - start line (np.ndarray)

        finish_line: [[x1, y1], [x2, y2]] - finish line (np.ndarray)

        center_line: [[x_i, y_i]. . . ] | None - center line of the map as a 2d array of points (np.ndarray) or None, optional (not present in all maps)

        racing_line: [[x_i, y_i]. . . ] | None - racing line of the map as a 2d array of points (np.ndarray) or None, optional (not present in all maps)

        cones: [[x1, y1, cls], [x2, y2, cls], ...] - cones of all colors (np.ndarray)
    """
    # Load json file
    with open(json_path, 'r') as f:
        map_dict = json.load(f)

    # Check if map is valid
    for key in ['car_position', 'yellow_cones', 'blue_cones', 'orange_cones', 'big_cones', 'start_line', 'finish_line']:
        assert key in map_dict, f"Missing key {key} in {json_path.name}"
        assert isinstance(map_dict[key], list), f"Key {key} in {json_path.name} is not a list"

    assert "car_heading" in map_dict, f"Missing key car_heading in {json_path.name}"
    assert isinstance(map_dict["car_heading"], float), f"Key car_heading in {json_path.name} is not a float"

    ret = {}

    # Load car start position and heading
    ret["car_pose"] = np.array([*map_dict['car_position'], np.deg2rad(map_dict["car_heading"])])

    # Load finish and start line
    ret["start_line"] = np.array(map_dict['start_line'])
    ret["finish_line"] = np.array(map_dict['finish_line'])

    # Load cones
    all_cones = np.empty((0, 3))
    for key, cone_cls in zip(["yellow_cones", "blue_cones", "orange_cones", "big_cones"], [ConeClasses.YELLOW, ConeClasses.BLUE, ConeClasses.ORANGE, ConeClasses.BIG]):
        if len(map_dict[key]) != 0:
            cones = np.array(map_dict[key])
            cones = np.hstack((cones, np.full((len(cones), 1), fill_value=cone_cls.value)))
            ret[key] = cones
            all_cones = np.vstack((all_cones, cones))
        else:
            ret[key] = np.empty((0, 3))

    ret["cones"] = all_cones

    # Load center line
    if "center_line" in map_dict and len(map_dict["center_line"]) != 0:
        ret["center_line"] = np.array(map_dict["center_line"])
    else:
        ret["center_line"] = None

    # Load racing line
    if "racing_line" in map_dict and len(map_dict["racing_line"]) != 0:
        ret["racing_line"] = np.array(map_dict["racing_line"])
    else:
        ret["racing_line"] = None

    if "speed_profile" in map_dict and len(map_dict["speed_profile"]) != 0:
        ret["speed_profile"] = np.array(map_dict["speed_profile"])
    else:
        ret["speed_profile"] = None

    return ret
