import numpy as np
import math
from config import car_params


def get_lookahead_point(lookahead, path):
    """
    Given piecewise linear function and distance, returns a point that is that distance away on the line
    Args:
      lookahead - distance constant
      path - Nx2 numpy array
    Ret:
      target - 2D point
    """
    if path.shape[0] == 0:
        return np.array([0., 0.])

    target = path[-1, :]

    cum_dist = 0.
    for i in range(0, path.shape[0] - 1):
        line = path[i + 1] - path[i]
        dist = np.linalg.norm(line)
        cum_dist += dist

        if cum_dist >= lookahead:
            last_piece_coef = (cum_dist - lookahead) / (dist + 1e-6)
            target = path[i + 1] - last_piece_coef * line
            break
    return target


def get_lookahead_idx(path, start_idx, lookahead_dist):
    lookahead_idx = start_idx
    cum_dist = 0.
    while cum_dist < lookahead_dist:
        cum_dist += np.linalg.norm(path[(lookahead_idx + 1) % path.shape[0]] - path[lookahead_idx % path.shape[0]])
        lookahead_idx += 1
    return lookahead_idx % path.shape[0]


def stanley_steering(path, lookahead_dist, speed, gain, lateran_gain):
    """
    Lateral steering controller
    Args:
      path - Nx2 numpy array representing the path the car should drive in car's coordinate system
      lookahead_dist - distance determining the lookahead_point on the path line
      speed - current speed of the car
      gain - controller parameter
      lateran_gain - controller parameter
      max_range - determines the output range of the controller to (-max_range, max_range)
    Ret:
      delta - steering wheel angle
      log_msg - dictionary containing internal values of the controller (for debugging purposes)
    """
    # Adjust damping for higher speeds
    if speed > 0:
        lookahead_dist = lookahead_dist * (1 + speed / 45)
        lookahead_dist = np.clip(lookahead_dist, 0.1, 8.0)
        damping_factor = 1 / (1 + speed / 15)  # Less damping at high speeds
    else:
        damping_factor = 1.0

    target = get_lookahead_point(lookahead_dist, path)
    lateral_target = get_lookahead_point(0.0, path)

    dx = target[0]
    dy = target[1]

    direction = math.atan2(dy, dx)
    if len(path) > 1:
        lat_offset = lateral_target[1]
    else:
        lat_offset = 0

    # More aggressive cornering at higher speeds
    if speed > 0.05:
        # Progressive corner boost that increases with speed
        speed_factor = (1.0 + speed / 10)  # Increases effect at higher speeds
        corner_boost = (1.0 + abs(lat_offset)) * speed_factor
        nonLinear = 5.0 * corner_boost * math.atan2(lateran_gain * lat_offset, speed) / np.pi * damping_factor
    else:
        nonLinear = 0

    # Keep aggressive base gain
    linear = gain * 1.6 * direction * damping_factor

    if np.linalg.norm(target) > 0.05:
        delta = linear + nonLinear
    else:
        delta = 0.0

    delta *= 180 / np.pi
    delta = np.clip(delta, car_params["min_steering_angle"], car_params["max_steering_angle"])

    log_message = {"linear": linear,
                   "non_linear": nonLinear,
                   "lateral_offset": lat_offset,
                   "target": target,
                   "direction": direction,
                   "delta": delta
                   }

    return delta, log_message
