import math
from pathlib import Path
import time
import numpy as np
import platform
import warnings
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import ConeClasses
from helpers.unit_conversions import denormalize_angle, wheel_rpm_to_mps
from config import car_params
from helpers.geometry import rotate_points, global_to_local, local_to_global
from helpers.physics_models import kinematic_model
from helpers.geometry import global_to_local, rotate_points
from helpers.cone_utils import filter_occluded_cones, load_map_from_json, filter_points_by_fov, apply_noise
from helpers.unit_conversions import steering_angle_to_wheel_angle, wheel_angle_to_steering_angle, normalize_angle
from config import ConeClasses
from helpers.pid import PID
from collections import deque, defaultdict


def find_lcm(a, b):
    return a * b // math.gcd(a, b)


class State():
    def __init__(self, map_filepath: str | Path, config: dict):
        self.config = config
        self.speed = 0.
        self.noisy_speed = 0.
        self.speed_set_point = 0.
        self.steering_angle = 0.
        self.controller = PID(**config["controller_gains"], min_output=-car_params["max_motor_torque"] * 4, max_output=car_params["max_motor_torque"] * 4)
        self.steering_angle_set_point = 0.
        self.car_pose = np.array([0., 0., 0.])  # x, y, yaw
        self.starting_pose = self.car_pose.copy()
        self.velocity = np.array([0., 0., 0.])  # vx, vy, yaw_rate
        self.steering_state = [0.0, 0.0]  # pos speed, acceleration
        self.steering_input_queue = deque(maxlen=9)
        self.steering_input_queue.extend([0.0] * 9)
        self.setpoint_history = defaultdict(list)
        self.actual_history = defaultdict(list)
        self.load_map(map_filepath)

    def update_state(self, timedelta: float) -> None:
        self.update_steering_angle(timedelta)
        torque = self.controller.step(self.speed_set_point, self.noisy_speed, timedelta)
        self.update_from_kinematic_model(torque, timedelta)
        self.check_cone_collisions()
        self.setpoint_history["speed"].append(self.speed_set_point)
        self.setpoint_history["steering_angle"].append(self.steering_angle_set_point)
        self.actual_history["speed"].append(self.noisy_speed)
        self.actual_history["steering_angle"].append(self.steering_angle)
        self.actual_history["pose"].append(self.car_pose.copy())
        self.actual_history["velocity"].append(self.velocity.copy())
        self.actual_history["torque"].append(torque)
        self.actual_history["noisy_speed"].append(self.noisy_speed)
        self.actual_history["cones_hit"].append(self.cones_hit.sum())

    def update_steering_angle(self, timedelta: float) -> None:
        steer_input = self.steering_input_queue.pop()
        self.steering_input_queue.append(self.steering_angle_set_point)
        self.steering_state[0] = np.clip(self.steering_state[0], car_params["min_wheel_angle"], car_params["max_wheel_angle"])
        acceleration = car_params["steering_k"] * steer_input - 2 * car_params["steering_zeta"] * car_params["steering_omega"] * self.steering_state[1] - car_params["steering_omega"] ** 2 * self.steering_state[0]
        acceleration = np.clip(acceleration, -car_params["steering_max_acceleration"], car_params["steering_max_acceleration"])
        self.steering_state[1] += acceleration * timedelta
        self.steering_state[1] = np.clip(self.steering_state[1], -car_params["steering_max_speed"], car_params["steering_max_speed"])
        self.steering_state[0] += self.steering_state[1] * timedelta
        self.steering_angle = self.steering_state[0]
        self.steering_angle = np.clip(self.steering_angle, car_params["min_wheel_angle"], car_params["max_wheel_angle"])

    def update_from_kinematic_model(self, torque: float, timedelta: float) -> None:
        force = torque * car_params["gear_ratio"] / car_params["wheel_radius"]
        acceleration = force / car_params["mass"]
        self.speed += acceleration * timedelta
        vx, vy, yaw_rate = kinematic_model(self.car_pose[2], self.speed, np.deg2rad(self.steering_angle))
        self.velocity = np.array([vx, vy, yaw_rate])
        self.car_pose += self.velocity * timedelta
        self.noisy_speed = self.speed + np.random.normal(0., wheel_rpm_to_mps(self.config["rpm_noise"]))

    def load_map(self, map_filepath: str | Path) -> None:
        self.map_dict = load_map_from_json(map_filepath)
        self.cones_world = self.map_dict["cones"]
        self.cones_hit = np.zeros(self.cones_world.shape[0], dtype=bool)
        match self.config["track_noise_type"]:
            case "uniform":
                noise = np.random.uniform(-1., 1., size=(self.cones_world.shape[0], 2)) * self.config["track_build_noise"]
            case "normal":
                noise = np.random.normal(0., self.config["track_build_noise"] / 3, size=(self.cones_world.shape[0], 2))
        self.cones_world[:, :2] += noise
        self.yellow_cones = self.cones_world[self.cones_world[:, 2] == ConeClasses.YELLOW]
        self.blue_cones = self.cones_world[self.cones_world[:, 2] == ConeClasses.BLUE]
        self.orange_cones = self.cones_world[self.cones_world[:, 2] == ConeClasses.ORANGE]
        self.big_cones = self.cones_world[self.cones_world[:, 2] == ConeClasses.BIG]
        self.start_line = self.map_dict["start_line"]
        self.finish_line = self.map_dict["finish_line"]
        self.car_pose = self.map_dict["car_pose"]
        self.start_pose = self.map_dict["car_pose"].copy()

    def get_detections(self):
        cones_local = self.cones_world.copy()
        cones_local[:, 0:2] = global_to_local(cones_local[:, 0:2], self.car_pose)
        cones_local = filter_points_by_fov(cones_local, self.config["max_dist"], self.config["fov"])
        cones_local = filter_occluded_cones(cones_local, self.config["occlusion_profile"])
        cones_local = apply_noise(cones_local, self.config, self.config["occlusion_profile"])
        return cones_local

    def check_cone_collisions(self) -> None:
        cones_local = self.cones_world.copy()
        cones_local[:, 0:2] = global_to_local(cones_local[:, 0:2], self.car_pose)
        cones_x = cones_local[:, 0]
        cones_y = cones_local[:, 1]
        width = car_params["wheel_base"]
        length = car_params["CoG_distance_from_end"]
        colliding_cones = (cones_x < length / 2) & (cones_x > -length / 2) & (cones_y < width / 2) & (cones_y > -width / 2)
        self.cones_hit[colliding_cones] = True

    def get_obs(self):
        obs = {
            "percep_data": self.get_detections(),
            "actual_speed": self.noisy_speed,
            "actual_steering_angle": (wheel_angle_to_steering_angle(self.steering_angle)) + np.random.normal(0., 0.1),
            "ins_position": rotate_points(self.car_pose[: 2].copy() - self.start_pose[: 2],
                                          -self.start_pose[2]),
            "ins_heading": np.rad2deg(normalize_angle(self.car_pose[2] - self.start_pose[2])),
            "velocity": self.velocity,
            "car_pose": self.car_pose,
            "ins_imu_gyro": np.array([0., 0., -1. * np.rad2deg(self.velocity[2])])
        }
        return obs

    def set_new_setpoints(self, steering_angle: float, speed: float) -> None:
        self.steering_angle_set_point = steering_angle_to_wheel_angle(steering_angle)
        self.speed_set_point = speed

    def is_within_roi(self, roi_thresh: float = 200):
        return np.linalg.norm(self.car_pose[:2] - self.start_pose[:2]) < roi_thresh


class StateRenderer:
    def __init__(self):
        self.fig, (self.global_ax, self.local_ax) = plt.subplots(1, 2, figsize=(20, 6))
        self.figure_size = self.fig.get_size_inches()
        self.map_ratio = 1.
        self.max_width = 19.0
        # styling
        self.fig.set_facecolor((0.6, 0.6, 0.55))
        self.local_ax.set_facecolor((0.5, 0.5, 0.5))
        self.global_ax.set_facecolor((0.5, 0.5, 0.5))
        self.local_ax.grid(alpha=0.6, which='major', color=(0.6, 0.6, 0.55))
        self.local_ax.grid(alpha=0.3, which='minor', color=(0.6, 0.6, 0.55))
        self.global_ax.grid(alpha=0.6, which='major', color=(0.6, 0.6, 0.55))
        self.global_ax.grid(alpha=0.3, which='minor', color=(0.6, 0.6, 0.55))

        # local ax setup
        self.local_ax.set_xlim(0., 24)
        self.local_ax.set_ylim(-12, 12)
        self.local_ax.set_xticks(np.arange(0, 24, 1), minor=True)
        self.local_ax.set_yticks(np.arange(-12, 12, 1), minor=True)
        self.local_ax.set_aspect('equal', adjustable='box')

        # global lines
        self.map_yc_line, = self.global_ax.plot([], [], '.', color='gold')
        self.map_bc_line, = self.global_ax.plot([], [], '.', color='blue')
        self.map_oc_line, = self.global_ax.plot([], [], '.', color='orange')
        self.map_big_line, = self.global_ax.plot([], [], '.', color='red')
        self.map_static_lines = [self.map_yc_line, self.map_bc_line, self.map_oc_line, self.map_big_line]

        self.glob_text = self.global_ax.text(0., 0., "", color=(0, 0, 0), verticalalignment='top')
        self.map_path_line, = self.global_ax.plot([], [], '-', color='red')
        self.map_car_pose, = self.global_ax.plot([], [], 'o', color=(1, 0.4, 0), markersize=7)
        self.map_front_wing, = self.global_ax.plot([], [], '-', color=(0.85, 0.35, 0))
        self.map_chassis, = self.global_ax.plot([], [], '-', color=(0.95, 0.38, 0))
        self.map_driver, = self.global_ax.plot([], [], '-', color=(0, 0, 0))
        self.map_FL_wheel, = self.global_ax.plot([], [], '-', color=(0, 0, 0))
        self.map_FR_wheel, = self.global_ax.plot([], [], '-', color=(0, 0, 0))
        self.map_RR_wheel, = self.global_ax.plot([], [], '-', color=(0, 0, 0))
        self.map_RL_wheel, = self.global_ax.plot([], [], '-', color=(0, 0, 0))
        self.map_rear_wing, = self.global_ax.plot([], [], '-', color=(1, 0.4, 0))
        self.map_floor, = self.global_ax.plot([], [], '-', color=(0.85, 0.35, 0))
        self.map_dynamic_lines = [self.map_rear_wing, self.map_driver, self.map_chassis, self.map_front_wing, self.map_floor, self.map_RR_wheel, self.map_RL_wheel, self.map_FL_wheel, self.map_FR_wheel, self.map_path_line]

        # local lines
        self.loc_yc_line, = self.local_ax.plot([], [], '.', color='gold')
        self.loc_bc_line, = self.local_ax.plot([], [], '.', color='blue')
        self.loc_oc_line, = self.local_ax.plot([], [], '.', color='orange')
        self.loc_big_line, = self.local_ax.plot([], [], '.', color='red')
        self.loc_car_pose, = self.local_ax.plot([], [], 'o', color=(1, 0.4, 0))
        self.loc_path_line, = self.local_ax.plot([], [], '-', color='red')
        self.loc_front_wing, = self.local_ax.plot([], [], '-', color=(0.85, 0.35, 0))
        self.loc_chassis, = self.local_ax.plot([], [], '-', color=(0.95, 0.38, 0))
        self.loc_driver, = self.local_ax.plot([], [], '-', color=(0, 0, 0))
        self.loc_FL_wheel, = self.local_ax.plot([], [], '-', color=(0, 0, 0))
        self.loc_FR_wheel, = self.local_ax.plot([], [], '-', color=(0, 0, 0))
        self.loc_floor, = self.local_ax.plot([], [], '-', color=(0.85, 0.35, 0))
        self.loc_static_lines = [self.loc_floor, self.loc_front_wing, self.loc_chassis, self.loc_driver]
        self.loc_dynamic_lines = [self.loc_path_line, self.loc_yc_line, self.loc_bc_line, self.loc_oc_line, self.loc_big_line, self.loc_FL_wheel, self.loc_FR_wheel]

        self.fig.canvas.mpl_connect('key_press_event', lambda event: exit(0) if event.key == 'e' else None)

    def create_car_parts(self, spacing):
        xpoints = np.arange(-0.2, 0.2, 0.19)
        ypoints = np.array([-0.12, 0.12])
        wheel = np.array([[xpoints[0], ypoints[1]], [xpoints[-1], ypoints[1]], [xpoints[-1], ypoints[0]], [xpoints[0], ypoints[0]]])
        for x in xpoints:
            for y in ypoints:
                wheel = np.append(wheel, np.array([[x, y]]), axis=0)

        xpoints = np.arange(-car_params["rear_axle"] + 0.3, car_params["front_axle"] - 0.4, spacing)
        ypoints = np.array([-car_params["side_axle"] - 0.15, 0, car_params["side_axle"] + 0.15])
        floor = np.array([[xpoints[0], ypoints[-1]], [xpoints[-1], ypoints[-1]], [xpoints[-1], ypoints[0]], [xpoints[0], ypoints[0]]])
        for x in xpoints:
            for y in ypoints:
                floor = np.append(floor, np.array([[x, y]]), axis=0)

        xpoints = np.arange(0.4, 0.75, spacing)
        ypoints = np.array([-car_params["side_axle"] - 0.12, 0, car_params["side_axle"] + 0.12])
        front_wing = np.array([[xpoints[0], ypoints[-1]], [xpoints[-1], ypoints[-1]], [xpoints[-1], ypoints[0]], [xpoints[0], ypoints[0]]])
        for x in xpoints:
            for y in ypoints:
                front_wing = np.append(front_wing, np.array([[x, y]]), axis=0)
        front_wing += np.array([car_params["front_axle"], 0])

        xpoints = np.arange(-0.3, 0.4, 0.25)
        ypoints = np.array([-car_params["side_axle"] + 0.1, car_params["side_axle"] - 0.1])
        rear_wing = np.array([[xpoints[0], ypoints[-1]], [xpoints[-1], ypoints[-1]], [xpoints[-1], ypoints[0]], [xpoints[0], ypoints[0]]])
        for x in xpoints:
            for y in ypoints:
                rear_wing = np.append(rear_wing, np.array([[x, y]]), axis=0)

        rear_wing -= np.array([car_params["rear_axle"], 0])

        xpoints = np.arange(-car_params["rear_axle"], car_params["front_axle"] + 0.65, spacing)
        ypoints = np.array([-car_params["side_axle"] + 0.32, 0, car_params["side_axle"] - 0.32])
        chassis = np.array([[xpoints[0], ypoints[-1]], [xpoints[-1], ypoints[-1]], [xpoints[-1], ypoints[0]], [xpoints[0], ypoints[0]]])
        for x in xpoints:
            for y in ypoints:
                chassis = np.append(chassis, np.array([[x, y]]), axis=0)

        xpoints = np.arange(-0.3, 0.2, spacing)
        ypoints = np.array([-car_params["side_axle"] + 0.45, car_params["side_axle"] - 0.45])
        driver = np.array([[xpoints[0], ypoints[-1]], [xpoints[-1], ypoints[-1]], [xpoints[-1], ypoints[0]], [xpoints[0], ypoints[0]]])
        for x in xpoints:
            for y in ypoints:
                driver = np.append(driver, np.array([[x, y]]), axis=0)

        self.car_parts = {'wheel': wheel, 'floor': floor, 'chassis': chassis, 'front_wing': front_wing, 'rear_wing': rear_wing, 'driver': driver}

    def set_state(self, state):
        self.state = state
        self.determine_map_bounds()
        self.draw_static_map()

    def determine_map_bounds(self):
        cones = self.state.cones_world.copy()
        cones[:, :2] = global_to_local(cones[:, :2], self.state.start_pose)
        offset = 5
        height = self.figure_size[1]

        self.x_size = cones[:, 0].max() - cones[:, 0].min() + 2 * offset
        self.y_size = cones[:, 1].max() - cones[:, 1].min() + 3 * offset
        self.determine_map_ratio()

        width = height + (height * self.map_ratio)
        if self.max_width < width:
            self.map_ratio *= (self.max_width - height) / (width - height)
            width = self.max_width

        self.fig.set_size_inches(width, height)
        self.figure_size = self.fig.get_size_inches()
        self.global_ax.set_xticks(np.arange(((cones[:, 0].min() - offset) // 25) * 25, ((cones[:, 0].max() + offset) // 25 + 1) * 25, 25)[1:])
        self.global_ax.set_xticks(np.arange(((cones[:, 0].min() - offset) // 5) * 5, ((cones[:, 0].max() + offset) // 5 + 1) * 5, 5)[1:], minor=True)
        self.global_ax.set_yticks(np.arange(((cones[:, 1].min() - offset) // 25) * 25, ((cones[:, 1].max() + offset * 2) // 25 + 1) * 25, 25)[1:])
        self.global_ax.set_yticks(np.arange(((cones[:, 1].min() - offset) // 5) * 5, ((cones[:, 1].max() + offset * 2) // 5 + 1) * 5, 5)[1:], minor=True)
        self.global_ax.set_xlim(cones[:, 0].min() - offset, cones[:, 0].max() + offset)
        self.global_ax.set_ylim(cones[:, 1].min() - offset, cones[:, 1].max() + offset * 2)
        self.glob_text.set_position((cones[:, 0].min() - offset + 0.5, cones[:, 1].max() + 2.0 * offset - 0.5))
        self.global_ax.set_aspect('equal', adjustable='box')

    def determine_map_ratio(self):
        height = self.figure_size[1]

        self.map_ratio = self.x_size / self.y_size
        width = height + (height * self.map_ratio)
        if self.figure_size[0] < width:
            self.map_ratio *= (self.figure_size[0] - height) / (width - height)

    def draw_static_map(self, extra_global_lines=[]):
        self.determine_map_ratio()

        gs = GridSpec(1, 2,
                      width_ratios=[self.map_ratio, 1],
                      right=1 - 0.2 / self.figure_size[0],
                      top=0.99,
                      left=0.45 / self.figure_size[0],
                      bottom=0.25 / self.figure_size[1],
                      wspace=1.2 / self.figure_size[0])

        axes = [self.global_ax, self.local_ax]
        for i in range(2):
            ax = axes[i]
            ax.set_position(gs[i].get_position(self.fig))

        obj_size = 0.30
        linewidth = 0.1
        map_bbox = self.global_ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        for line in self.map_dynamic_lines + self.map_static_lines:
            if line.get_marker() == '.':
                line.set_markersize(np.min([obj_size / self.x_size * map_bbox.width, obj_size / self.y_size * map_bbox.height]) * 10000 / 72)
            else:

                line.set_linewidth(np.min([linewidth / self.x_size * map_bbox.width, linewidth / self.y_size * map_bbox.height]) * 10000 / 72)

        local_bbox = self.local_ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        local_size = self.local_ax.get_ylim()[1] - self.local_ax.get_ylim()[0]

        for line in self.loc_dynamic_lines + self.loc_static_lines:
            if line.get_marker() == '.':
                m_size = 100 / 72 * np.max([1, obj_size / local_size * np.min([local_bbox.height, local_bbox.width]) * 100])
                line.set_markersize(m_size)
            else:
                l_width = 100 / 72 * np.max([1, 0.05 / local_size * np.min([local_bbox.height, local_bbox.width]) * 100])
                line.set_linewidth(l_width)

        # reset data buffers
        for line in self.loc_dynamic_lines + self.map_dynamic_lines + self.loc_static_lines + self.map_static_lines:
            line.set_data([], [])

        self.glob_text.set_text("")
        self.map_car_pose.set_data([], [])
        plt.pause(0.0001)

        # draw static global map
        cones = self.state.cones_world.copy()
        cones[:, :2] = global_to_local(cones[:, :2], self.state.start_pose)

        self.map_yc_line.set_data(*cones[cones[:, 2] == ConeClasses.YELLOW][:, 0:2].T)
        self.map_bc_line.set_data(*cones[cones[:, 2] == ConeClasses.BLUE][:, 0:2].T)
        self.map_oc_line.set_data(*cones[cones[:, 2] == ConeClasses.ORANGE][:, 0:2].T)
        self.map_big_line.set_data(*cones[cones[:, 2] == ConeClasses.BIG][:, 0:2].T)
        for line in self.map_static_lines:
            self.global_ax.draw_artist(line)
        for line in extra_global_lines:
            self.global_ax.draw_artist(line)
        self.global_static_background = self.fig.canvas.copy_from_bbox(self.global_ax.bbox)
        self.create_car_parts(0.12)
        # draw static local map
        self.loc_car_pose.set_data([0.], [0.])
        self.loc_front_wing.set_data(self.car_parts['front_wing'][:, 0], self.car_parts['front_wing'][:, 1])
        self.loc_chassis.set_data(self.car_parts['chassis'][:, 0], self.car_parts['chassis'][:, 1])
        self.loc_driver.set_data(self.car_parts['driver'][:, 0], self.car_parts['driver'][:, 1])
        self.loc_floor.set_data(self.car_parts['floor'][:, 0], self.car_parts['floor'][:, 1])
        for line in self.loc_static_lines:
            self.local_ax.draw_artist(line)
        self.local_static_background = self.fig.canvas.copy_from_bbox(self.local_ax.bbox)
        self.create_car_parts(0.25)

    def render_state(self, text, path=None, draw_extra=None):
        if np.any(self.figure_size != self.fig.get_size_inches()):
            self.draw_static_map()
            self.figure_size = self.fig.get_size_inches()

        start = time.time()
        self.fig.canvas.restore_region(self.global_static_background)
        self.fig.canvas.restore_region(self.local_static_background)
        obs = self.state.get_obs()

        # plot text fast
        self.glob_text.set_text(text)
        self.global_ax.draw_artist(self.glob_text)

        # global state
        if draw_extra is not None:
            draw_extra()

        wheel_angle = np.deg2rad(self.state.steering_angle)
        fr_wheel = rotate_points(self.car_parts["wheel"].copy(), wheel_angle) - np.array([-car_params["front_axle"], car_params["side_axle"]])
        fl_wheel = rotate_points(self.car_parts["wheel"].copy(), wheel_angle) - np.array([-car_params["front_axle"], -car_params["side_axle"]])

        heading = denormalize_angle(np.deg2rad(obs["ins_heading"]))
        if path is not None:
            glob_path = local_to_global(path.copy(), np.array([*obs["ins_position"], heading]))
            self.map_path_line.set_data(glob_path[:, 0], glob_path[:, 1])
        self.map_car_pose.set_data(obs["ins_position"][:2].reshape(-1, 1))

        car_pose = np.array([*obs["ins_position"], heading])

        rear_wing = local_to_global(self.car_parts["rear_wing"].copy(), car_pose)
        self.map_rear_wing.set_data(rear_wing[:, 0], rear_wing[:, 1])
        chassis = local_to_global(self.car_parts["chassis"].copy(), car_pose)
        self.map_chassis.set_data(chassis[:, 0], chassis[:, 1])
        driver = local_to_global(self.car_parts["driver"].copy(), car_pose)
        self.map_driver.set_data(driver[:, 0], driver[:, 1])
        front_wing = local_to_global(self.car_parts["front_wing"].copy(), car_pose)
        self.map_front_wing.set_data(front_wing[:, 0], front_wing[:, 1])
        floor = local_to_global(self.car_parts["floor"].copy(), car_pose)
        self.map_floor.set_data(floor[:, 0], floor[:, 1])
        glob_fr_wheel = local_to_global(fr_wheel.copy(), car_pose)
        self.map_FR_wheel.set_data(glob_fr_wheel[:, 0], glob_fr_wheel[:, 1])
        glob_fl_wheel = local_to_global(fl_wheel.copy(), car_pose)
        self.map_FL_wheel.set_data(glob_fl_wheel[:, 0], glob_fl_wheel[:, 1])
        rr_wheel = local_to_global(self.car_parts["wheel"] - np.array([car_params["rear_axle"], car_params["side_axle"]]), car_pose)
        self.map_RR_wheel.set_data(rr_wheel[:, 0], rr_wheel[:, 1])
        rl_wheel = local_to_global(self.car_parts["wheel"] - np.array([car_params["rear_axle"], -car_params["side_axle"]]), car_pose)
        self.map_RL_wheel.set_data(rl_wheel[:, 0], rl_wheel[:, 1])
        for line in self.map_dynamic_lines[::-1]:
            self.global_ax.draw_artist(line)

        # local state
        cones = obs["percep_data"]
        self.loc_yc_line.set_data(*cones[cones[:, 2] == ConeClasses.YELLOW][:, 0:2].T)
        self.loc_bc_line.set_data(*cones[cones[:, 2] == ConeClasses.BLUE][:, 0:2].T)
        self.loc_oc_line.set_data(*cones[cones[:, 2] == ConeClasses.ORANGE][:, 0:2].T)
        self.loc_big_line.set_data(*cones[cones[:, 2] == ConeClasses.BIG][:, 0:2].T)
        self.loc_FL_wheel.set_data(fl_wheel[:, 0], fl_wheel[:, 1])
        self.loc_FR_wheel.set_data(fr_wheel[:, 0], fr_wheel[:, 1])
        if path is not None:
            self.loc_path_line.set_data(path[:, 0], path[:, 1])
        for line in self.loc_dynamic_lines:
            self.local_ax.draw_artist(line)

        self.fig.canvas.blit(self.global_ax.bbox)
        self.fig.canvas.blit(self.local_ax.bbox)
        self.fig.canvas.flush_events()
        # print(f"fps: {1/(time.time()-start)}")

    def wait_for_close(self):
        plt.waitforbuttonpress()

    def close(self):
        plt.close(self.fig)


def make_simulation_object(
        state: State,
        mission: object,
        state_fps: int = 90,
        mission_fps: int = 30,
):
    lcm_val = find_lcm(mission_fps, state_fps)
    mission_rate = lcm_val // mission_fps
    state_rate = lcm_val // state_fps
    i = 0
    mission_time = 0.
    while True:
        if i % state_rate == 0:
            state.update_state(1 / state_fps)
        if i % mission_rate == 0:
            obs = state.get_obs()
            mission_action = mission.loop(obs, mission_time)
            state.set_new_setpoints(mission_action[2], mission_action[1])
            mission_time += 1 / mission_fps
            yield mission_time, obs, mission_action
        i += 1


def history_to_csv(state: State, destination: str | Path):
    import pandas as pd
    time = np.arange(len(state.setpoint_history["speed"])) * 1 / 90.
    state.actual_history["time"] = list(time)
    df = pd.DataFrame(state.actual_history | state.setpoint_history)
    df.to_csv(destination, index=False)


def plot_state_summary_and_wait(state: State, finish_time: float, success: bool):
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    fig.subplots_adjust(top=0.7, hspace=0.5)
    fig.set_facecolor((0.6, 0.6, 0.55))
    for axis in ax.flatten():
        axis.grid(alpha=0.6, which='major', color=(0.6, 0.6, 0.55))
        axis.grid(alpha=0.3, which='minor', color=(0.6, 0.6, 0.55))
        axis.set_facecolor((0.5, 0.5, 0.5))
    tit = f"You finished in {finish_time:.2f} seconds!" if success else f"Car went off track!"
    fig.suptitle(tit + " Here is a summary of your run.", fontsize=16, fontweight='bold', y=0.99)
    # Speed
    time = np.arange(len(state.setpoint_history["speed"])) * 1 / 90.
    ax[0, 0].plot(time, state.setpoint_history["speed"], label="Speed Setpoint", color="blue")
    ax[0, 0].plot(time, state.actual_history["speed"], label="Actual Speed", color="red")
    ax[0, 0].legend(loc="lower left", facecolor="black", labelcolor="white")
    ax[0, 0].set_title("Speed")
    ax[0, 0].set_xlabel("Time (seconds)")
    ax[0, 0].set_ylabel("Speed (m/s)")
    ax[0, 0].set_xticks(time[::180], labels=[f"{int(t)}" for t in time[::180]])
    # Torque
    ax[0, 1].plot(time, state.actual_history["torque"], label="Torque", color="blue")
    ax[0, 1].set_title("Torque")
    ax[0, 1].set_xlabel("Time (seconds)")
    ax[0, 1].set_ylabel("Torque (Nm)")
    ax[0, 1].set_xticks(time[::180], labels=[f"{int(t)}" for t in time[::180]])
    # Steering Angle
    ax[1, 0].plot(time, state.setpoint_history["steering_angle"], label="Steering Angle Setpoint", color="blue")
    ax[1, 0].plot(time, state.actual_history["steering_angle"], label="Actual Steering Angle", color="red")
    ax[1, 0].legend(loc="lower left", facecolor="black", labelcolor="white")
    ax[1, 0].set_title("Steering Angle")
    ax[1, 0].set_xlabel("Time (seconds)")
    ax[1, 0].set_ylabel("Steering Angle (deg)")
    ax[1, 0].set_xticks(time[::180], labels=[f"{int(t)}" for t in time[::180]])
    # Car Pose
    car_poses = np.array(state.actual_history["pose"])
    for i, p in enumerate(car_poses):
        if i % 30 == 0:
            ax[1, 1].arrow(p[0], p[1], np.cos(p[2]), np.sin(p[2]), head_width=0.5, head_length=0.5, fc='black', ec='black')
    ax[1, 1].plot(car_poses[:, 0], car_poses[:, 1], color="blue")
    ax[1, 1].set_title("Car Pose")
    ax[1, 1].set_xlabel("X (m)")
    ax[1, 1].set_ylabel("Y (m)")
    ok_cones = state.cones_world[~state.cones_hit]
    hit_cones = state.cones_world[state.cones_hit]
    ax[1, 1].scatter(ok_cones[:, 0], ok_cones[:, 1], c="green", s=8, edgecolors='black', linewidth=0.5)
    ax[1, 1].scatter(hit_cones[:, 0], hit_cones[:, 1], c="red", s=8, edgecolors='black', linewidth=0.5)
    fig.tight_layout(pad=2.5, h_pad=5.0)
    try:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    except:
        warnings.warn("Could not maximize window. Please maximize manually.")
    fig.waitforbuttonpress()
    plt.close(fig)
