import numpy as np
from config import car_params


def kinematic_model(yaw, speed, delta_f):
    """
    Kinematic model of the car in global coordinates
    Args:
        yaw: angle of the car in radians
        speed: speed of the car in m/s
        delta_f: steering angle of the car in radians
    Returns:
        der: derivative of the state vector
            dx: x component of the velocity in global coordinates
            dy: y component of the velocity in global coordinates
            d_yaw: yaw rate in rad/s
    """
    lr = car_params["rear_axle"]
    wheel_base = car_params["wheel_base"]
    beta = np.arctan(lr * np.tan(delta_f) / wheel_base)
    d_yaw = speed * (np.tan(delta_f) * np.cos(beta) / wheel_base)
    dx = speed * np.cos(beta + yaw)
    dy = speed * np.sin(beta + yaw)
    der = [dx, dy, d_yaw]
    return der
