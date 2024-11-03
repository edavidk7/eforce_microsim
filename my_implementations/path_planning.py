import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay

from config import ConeClasses


FILLING_CONES_DISTANCE = 3.5
MOVEMENT_DIRECTION = 0


class PathPlanner:
    def __init__(self):
        pass

    def find_path(self, cones):
        cones = cones if cones is not None else np.zeros((0, 3))

        # Filter out cones that are not blue or yellow
        mask = np.isin(cones[:, 2], [ConeClasses.BLUE, ConeClasses.YELLOW])
        cones = cones[mask]

        blue = cones[cones[:, 2] == ConeClasses.BLUE]
        yellow = cones[cones[:, 2] == ConeClasses.YELLOW]

        # no cones, return a straight line
        if len(blue) == 0 and len(yellow) == 0:
            return PathPlanner._straight()

        # mostly one color, fill in the missing cones
        if (len(blue) <= 1 < len(yellow)) or (len(yellow) <= 1 < len(blue)):
            blue, yellow = PathPlanner.fill_missing(blue, yellow)
            cones = np.vstack((blue, yellow))

        # not enough cones for triangulation
        if len(cones) < 4:

            if len(blue) > 0 and len(yellow) > 0:
                return PathPlanner._middle_point(blue[0, :2], yellow[0, :2]).reshape(1, 2)

            return PathPlanner._straight()

        return self._delaunay_triangulation(cones)

    @staticmethod
    def _interpolate_cones(cones, num_points):
        """
        Min 3 cones are required to interpolate
        NOT USED RIGHT NOW
        """
        cone_class = cones[0, 2]

        dists = np.cumsum(np.sqrt(np.sum(np.diff(cones, axis=0) ** 2, axis=1)))
        dists = np.insert(dists, 0, 0)

        spline_x = CubicSpline(dists, cones[:, 0])
        spline_y = CubicSpline(dists, cones[:, 1])

        interpolated_dists = np.linspace(0, dists[-1], num_points)
        interpolated_x = spline_x(interpolated_dists)
        interpolated_y = spline_y(interpolated_dists)

        return np.vstack((interpolated_x, interpolated_y, np.full(num_points, cone_class))).T

    @staticmethod
    def _straight():
        return np.array([[1, 0]]).astype(float)

    @staticmethod
    def _delaunay_triangulation(points):
        tri = Delaunay(points[:, :2])
        path_points = []

        for simplex in tri.simplices:
            triangle = points[simplex]

            # Only consider triangles that span over the path
            if PathPlanner._spans_over_the_path(triangle):

                # Add midpoints of the triangle to the path
                for midpoint in PathPlanner._get_midpoints(triangle):
                    path_points.append(midpoint)

        if len(path_points) == 0:
            return np.array([[1, 1]]).astype(float)

        return PathPlanner._sort_points(np.array(path_points))

    @staticmethod
    def _spans_over_the_path(triangle):
        # One cone must have a different class than the others
        return not np.all(triangle[:, 2] == triangle[0, 2])

    @staticmethod
    def _middle_point(point1, point2):
        return (point1 + point2) / 2

    @staticmethod
    def _get_midpoints(triangle):
        for i in range(3):
            if triangle[i, 2] != triangle[(i + 1) % 3, 2]:
                yield PathPlanner._middle_point(triangle[i, :2], triangle[(i + 1) % 3, :2])

    @staticmethod
    def _sort_points(points):
        # Sort cones iteratively by finding the closest cone to the last cone, starting with [0, 0]
        sorted_points = []
        while len(points) > 0:
            if len(sorted_points) == 0:
                last_cone = np.array([0, 0])
            else:
                last_cone = sorted_points[-1]
            dists = np.linalg.norm(points - last_cone, axis=1)
            closest_cone_idx = np.argmin(dists)
            sorted_points.append(points[closest_cone_idx])
            points = np.delete(points, closest_cone_idx, axis=0)

        return np.array(sorted_points)

    @staticmethod
    def fill_missing(B, Y):
        """
        Copied the implementation from the helpers/path_planning.py
        (fixed a bug in the vec shape - the last dimension indicating the cone class had to be removed)
        """
        yellow = True if len(Y) > len(B) else False
        to_fill = B if yellow else Y
        full = Y if yellow else B

        # sort the full array by the movement direction
        full = full[np.argsort(full[:, MOVEMENT_DIRECTION])]

        if len(to_fill) == 0:
            to_fill = to_fill.reshape(-1, full.shape[1])

        for idx in range(len(to_fill), len(full)):

            if idx == 0:
                vec = full[1, :2] - full[0, :2]
                vec = vec / np.linalg.norm(vec)
                rotation_matrix = np.array([[0, -1], [1, 0]]) if yellow else np.array([[0, 1], [-1, 0]])
                vec = np.matmul(rotation_matrix, vec)

            elif idx == len(full) - 1:
                vec = full[-1, :2] - full[-2, :2]
                vec = vec / np.linalg.norm(vec)
                rotation_matrix = np.array([[0, -1], [1, 0]]) if yellow else np.array([[0, 1], [-1, 0]])
                vec = np.matmul(rotation_matrix, vec)

            else:
                vec1 = full[idx - 1, :2] - full[idx, :2]
                vec2 = full[idx + 1, :2] - full[idx, :2]
                vec1 = vec1 / (np.linalg.norm(vec1) + 1e-6)
                vec2 = vec2 / (np.linalg.norm(vec2) + 1e-6)
                angle = np.arccos(np.dot(vec1, vec2)) / 2

                if yellow:
                    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                else:
                    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])

                vec = np.matmul(rotation_matrix, vec2)

            # compute the final vectors
            final = full[idx, :2] + vec * FILLING_CONES_DISTANCE
            # Append cone class to vec
            final = np.append(final, [ConeClasses.BLUE if yellow else ConeClasses.YELLOW])
            to_fill = np.vstack((to_fill, final))

        if yellow:
            return to_fill, full
        else:
            return full, to_fill
