import numpy as np
import cv2
from typing import List, Tuple

# Geometry functions for converting between global and local coordinates
# and rotation calculations


def quad_to_center(d, e, f):
    x0 = -d / 2
    y0 = -e / 2
    r = np.sqrt(x0**2 + y0**2 - f)
    return np.array((x0, y0, r))


def fit_circle_nhom(X):
    X = X.T  # shape (2, N)
    ones = np.ones((1, X.shape[1]))
    A = np.vstack((X, ones))
    b = -np.sum(X**2, axis=0)
    sol = np.linalg.lstsq(A.T, b, rcond=None)
    return sol[0]


def rotation_matrix_from_vectors(vec_A, vec_B):
    """
    Find the rotation matrix that aligns vec_A to vec_B
    """
    a, b = vec_A / np.linalg.norm(vec_A), vec_B / np.linalg.norm(vec_B)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix, np.arccos(c), v  # Rotation matrix, angle, axis


def rotmat(ang):
    """Create a rotation matrix by ang radians"""
    s = np.sin(ang)
    c = np.cos(ang)
    return np.array([[c, -s], [s, c]])


def rotmat_x(theta):
    """Create a rotation matrix around the x-axis by theta radians"""
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotmat_y(theta):
    """Create a rotation matrix around the y-axis by theta radians"""
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotmat_z(theta):
    """Create a rotation matrix around the z-axis by theta radians"""
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def global_to_local(points, pose) -> np.ndarray:
    """Convert points from global to local coordinates
    From standard (x,y) to (x',y') where x' is forward and y' is left

    Args:
        points (np.ndarray): Points to be converted
        pose (np.ndarray): Position of the car in global coordinates (x,y,yaw)

    Returns:
        np.ndarray: Converted points
    """
    pos = pose[:2]
    ori = pose[2]
    R = np.array([[np.cos(ori), -np.sin(ori)],
                  [np.sin(ori), np.cos(ori)]])
    points[:, :2] -= pos
    points[:, :2] = (R.T @ points[:, :2].T).T
    return points


def local_to_global(points, pose) -> np.ndarray:
    """Convert points from local to global coordinates
    From (x',y') where x' is forward and y' is left to standard (x,y)

    Args:
        points (np.ndarray): Points to be converted
        pose (np.ndarray): Position of the car in global coordinates (x,y,yaw)

    Returns:
        np.ndarray: Converted points
    """
    pos = pose[:2]
    ori = pose[2]
    R = np.array([[np.cos(ori), -np.sin(ori)],
                  [np.sin(ori), np.cos(ori)]])

    points[:, :2] = (R @ points[:, :2].T).T
    points[:, :2] += pos
    return points


def rotate_points(points, angle) -> np.ndarray:
    """Rotate points around the origin

    Args:
        points (np.ndarray): Points to be rotated
        angle (float): Angle in radians

    Returns:
        np.ndarray: Rotated points
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    points = (R @ points.T).T
    return points


def angle_to_unit_vector(angle) -> np.ndarray:
    """Get a unit vector rotated by angle

    Args:
        angle (float): Angle in radians

    Returns:
        np.ndarray: Rotated unit vector
    """
    base = [1., 0.]
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])

    rotated_vec = R @ base
    return rotated_vec


def to_homogeneous(points):
    """Turns a set of points into homogeneous coordinates (adds a column of ones)

    Args:
        points (np.ndarray): points of Nx(D) shape
    Returns:
        points_hom (np.ndarray): Nx(D+1) shape
    """
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1))))

    return points_hom


def homography_project(H, points):
    """
    Args:
        H (np.ndarray): 3x3 homography matrix
        points (np.ndarray): Nx2 points
    Returns:
        projected (np.ndarray): Nx2 projected points
    """

    # image points to homogeneous coordinates
    points_hom = to_homogeneous(points)

    projected = H @ points_hom.T
    projected /= projected[2, :]
    projected = projected[:2, :].T
    return projected


def compute_angle(p1, p2, p3):
    """
    Computes the angle(in degrees) between three points.
    """
    # Create vectors from points
    v1 = p1 - p2
    v2 = p3 - p2

    # Compute dot product
    dot_product = np.dot(v1, v2)

    # Compute the magnitudes of vectors
    v1_magnitude = np.sqrt(np.sum(v1**2))
    v2_magnitude = np.sqrt(np.sum(v2**2))

    # Compute angle in radians
    angle_rad = np.arccos(dot_product / (v1_magnitude * v2_magnitude))

    cross_product = np.cross(v1, v2)

    # Correct the sign of the angle based on the direction of rotation
    if cross_product > 0:
        angle_rad *= -1

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    if angle_deg > 0.:
        new_angle = 180. - angle_deg
    else:
        new_angle = -(180 - abs(angle_deg))

    return new_angle


def compute_homography(image_points, world_points, method=0):
    """
    Args:
      image_points(np.ndarray): Nx2 points
      world_points(np.ndarray): Nx2 points
    Returns:
      H - homography matrix
      mask - mask of inliers
    """
    H, mask = cv2.findHomography(
        image_points,
        world_points,
        method=method
    )

    return H, mask

def image_to_bev(image, H, out_shape=(1280, 720), occlusion_profile=[0,20,-10,10]):
    output_height, output_width = out_shape
    meters_x_min, meters_x_max, meters_y_min, meters_y_max = occlusion_profile
    scale_x = output_width / (meters_x_max - meters_x_min)
    scale_y = output_height / (meters_y_max - meters_y_min)
    offset_x = -meters_x_min * scale_x
    offset_y = -meters_y_min * scale_y

    scale_translate_matrix = np.array([
        [scale_x, 0, offset_x],
        [0, -scale_y, offset_y],
        [0, 0, 1]
    ])
    H_scaled = scale_translate_matrix @ H
    return cv2.warpPerspective(image, H_scaled, (output_width, output_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

def orthogonal_projection(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0: return line_start  # Avoid division by zero if line_start == line_end
    projection_factor = np.dot(point_vec, line_vec) / line_len
    projection = line_start + projection_factor * line_vec
    return projection
