import numpy as np
from bros.configs.car_params import car_params


def get_earth_radius_at_pos(lat, sea_level=0.0):
    """
    Calculate the earth radius at the current origin given by latitude

    Args:
        lat (np.float64): latitude of current origin 
        sea_level (np.float64, optional): sea_level of the current position. Defaults to 0.
    Returns:
        np.float64: total earth radius at current position
    """
    #  Earth eccentricity e
    e = np.array(0.0818, dtype=np.float64)
    # Earth diameter at equator
    r_equator = np.array(6378000, dtype=np.int64)
    base_radius = r_equator * np.sqrt(
        (1 - (2 * e ** 2 - e ** 4) * np.sin(np.deg2rad(lat)) ** 2) / (1 - e ** 2 * np.sin(np.deg2rad(lat)) ** 2))
    return base_radius + sea_level


def lat_lon_to_meter_x_y(lat_lon_arr: np.ndarray, earth_radius: float, lat_lon_origin: np.ndarray) -> np.ndarray:
    """
    Converts the current position in degrees to position in meters based on origin array and earth radius at origin
    Args:
        lat_lon_arr (np.ndarray[np.float64, np.float64]): current position lat lon
        earth_radius (np.float64): earth radius at origin
        lat_lon_origin (np.ndarray[np.float64, np.float64]): lat lon at origin
    Returns:
        np.ndarray[np.float64, np.float64]: current position in meters from origin, order (x,y)
    """
    meter_yx = earth_radius * np.tan(np.deg2rad(lat_lon_arr - lat_lon_origin))
    return meter_yx[::-1]


def meter_x_y_to_lat_lon(meter_xy, earth_radius, lat_lon_origin):
    """
    Converts the current position in meters to position in degrees based on origin array and earth radius at origin
    Args:
        meter_xy (np.ndarray[np.float64, np.float64]): current position in meters from origin, order (x,y)
        earth_radius (np.float64): earth radius at origin
        lat_lon_origin (np.ndarray[np.float64, np.float64]): lat lon at origin
    Returns:
        np.ndarray[np.float64, np.float64]: current position lat lon
    """
    lat_lon_arr = lat_lon_origin + np.rad2deg(np.arctan(meter_xy[::-1] / earth_radius))
    return lat_lon_arr


def wheel_rpm_to_mps(wheel_rpm):
    """
    Converts wheel rpm to meters per second
    Args:
        wheel_rpm (np.float64): wheel rpm
    Returns:
        np.float64: meters per second
    """
    return wheel_rpm * 1 / (60 * car_params["gear_ratio"]) * 2 * car_params["wheel_radius"] * np.pi


def mps_to_wheel_rpm(mps):
    """
    Converts meters per second to wheel rpm
    Args:
        mps (np.float64): meters per second
    Returns:
        np.float64: wheel rpm
    """
    return mps * (60 * car_params["gear_ratio"]) * 1 / (2 * car_params["wheel_radius"] * np.pi)


def wheel_ang_speed_to_rpm(omega):
    return car_params["gear_ratio"] * 60 * omega / (2 * np.pi)


def wheel_ang_speed_to_mps(omega):
    return omega * car_params["wheel_radius"]


def calculate_steer_distribution(steering_angle, car_speed):
    """
    Calculates the distribution of the steering angle to the wheels
    Args:
        steering_angle (np.float64): steering angle in degrees
        car_speed (np.float64): car speed in m/s
    Returns:
        np.ndarray[np.float64, np.float64]: distribution of the steering angle to the wheels
    """
    # WIP
    return None


def steering_angle_to_wheel_angle(steering_wheel_angle: float) -> float:
    """Convert steering wheel angle to wheel angle

    Args:
        steering_wheel_angle (float): steering wheel angle in degrees

    Returns:
        float: wheel angle in degrees
    """
    return steering_wheel_angle / car_params["steering_ratio"]


def wheel_angle_to_steering_angle(wheel_angle: float) -> float:
    """Convert wheel angle to steering wheel angle

    Args:
        wheel_angle (float): wheel angle in degrees

    Returns:
        float: steering wheel angle in degrees
    """
    return wheel_angle * car_params["steering_ratio"]


def calculate_downforce(speed: float) -> float:
    """
    Calculate the downforce based on the speed
    Args:
        speed (float): speed in m/s

    Returns:
        float: downforce in N
    """
    dynamic_pressure = 0.5 * car_params["air_density"] * speed ** 2
    downforce = dynamic_pressure * car_params["frontal_area"] * car_params["drag_coefficient"]
    return downforce


def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi).

    Parameters:
        angle (float): The angle to normalize in radians.

    Returns:
        float: The normalized angle in the range [-pi, pi).
    """
    # Use modulo to bring the angle within [0, 2*pi)
    angle = angle % (2 * np.pi)

    # Map the angle to [-pi, pi)
    if angle >= np.pi:
        angle -= 2 * np.pi

    return angle


def denormalize_angle(angle):
    """
    Denormalize an angle to the range [0, 2*pi).

    Parameters:
        angle (float): The angle to denormalize in radians.

    Returns:
        float: The denormalized angle in the range [0, 2*pi).
    """
    # Use modulo to bring the angle within [0, 2*pi)
    angle = angle % (2 * np.pi)

    return angle
