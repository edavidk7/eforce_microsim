import numpy as np
from config import ConeClasses


class PathPlanning(object):
    def __init__(self, start_point, clockwise=-1, filling_cones_distance=3.5, movement_direction="x", debugging=False):
        """
        :params start_point: numpy.array with coordinates of starting point
        :param clockwise: direction in which we want to find path
                            (if start_point[1]==0 => direction is -1 !!!)

        """
        self.sorted_blue_cones = []
        self.sorted_yellow_cones = []
        self.pairings = set()
        self.start_points = [start_point]
        self.clockwise = clockwise
        # parameters of normal line
        self.k = 0
        self.c = None
        self.k_past = 0
        self.filling_cones_distance = filling_cones_distance
        assert movement_direction in ["x", "y"]
        self.movement_direction = int(movement_direction == "y")
        # "y"->1 cones[:,1], "x"->0 cones[:,0]
        self.debugging = debugging
        if self.debugging:
            self.ks = []
            self.cs = []
            self.direction_changes = []

    def reset(self, start_point, clockwise=-1):
        self.sorted_blue_cones = []
        self.sorted_yellow_cones = []
        self.pairings = set()
        self.start_points = [start_point]
        self.clockwise = clockwise
        # parameters of normal line
        self.k = 0
        self.c = None
        self.k_past = 0
        if self.debugging:
            self.ks = []
            self.cs = []
            self.direction_changes = []

    def is_already_added(self, point):
        np.all(np.isin(point, self.sorted_yellow_cones))
        return np.all(np.isin(point, self.sorted_yellow_cones)) or np.all(np.isin(point, self.sorted_blue_cones))

    def points_above_normal(self, points):
        return points[
            np.sign((points[:, 1] - self.k * points[:, 0] - self.c)
                    ) == np.sign(self.k + 1e-30) * self.clockwise
        ]

    def find_closest_one(self, points):
        closest_index = np.argmin(np.linalg.norm(
            points - self.start_points[-1], axis=1))
        closest_cone = points[closest_index]
        return closest_cone

    def calculate_center(self, pointB, pointY):
        return np.array([(pointB[0] - pointY[0]) / 2 + pointY[0], (pointB[1] - pointY[1]) / 2 + pointY[1]])

    def return_stack(self, object_name):
        if object_name == "yellow cones":
            return np.vstack(self.sorted_yellow_cones)
        elif object_name == "blue cones":
            return np.vstack(self.sorted_blue_cones)
        elif object_name == "centers":
            return np.vstack(self.start_points)

    def find_line_parameters(self, pointB, pointY, normal=True):
        k = (pointB[1] - pointY[1]) / (pointB[0] - pointY[0] + 1e-10) + 1e-10
        self.k_past = self.k
        self.k = -1 / k if normal else k
        c = pointB[1] - self.k * pointB[0]
        self.c = c

    def check_direction(self):
        # and not(self.auxiliary_variable):
        if self.k is not None and (self.k - self.k_past) == 0:
            self.clockwise = -self.clockwise

    def find_next_center(self, pointsB, pointsY, step=None, verbose=True):
        self.find_line_parameters(self.start_points[-1], self.start_points[-2])
        self.check_direction()

        if self.debugging:
            self.ks.append(self.k)
            self.cs.append(self.c)
            self.direction_changes.append(self.clockwise)

        B_hat = self.points_above_normal(pointsB)
        Y_hat = self.points_above_normal(pointsY)
        # set_trace()
        b = self.find_closest_one(B_hat)
        y = self.find_closest_one(Y_hat)

        if not self.is_already_added(b):
            self.sorted_blue_cones.append(b)
        if not self.is_already_added(y):
            self.sorted_yellow_cones.append(y)

        s = self.calculate_center(b, y)
        pairing = (*b, *y)
        if not pairing in self.pairings:
            self.start_points.append(s)
            self.pairings.add(pairing)

    def find_path(self, B, Y, n_steps, verbose=False):
        """
        Iterative path finding procedure
        In:
            B            set of all detected blue cones, [np.array]
            Y            set of all detected yellow cones, [np.array]
            n_steps      number of steps, [int]
                            (if more steps then cones then loop stop itself,
                             but quality of predictions may degrade)
            verbose      verbose or not. [bool: True/False]

        Out:
            self         pointer to OldPathPlanning root object


        -> fill_missing is only available in single frame path prediction (SFPP) scenario

        """

        if B.size == 0 or Y.size == 0:
            """
            fill_missing suppose that there are only two known cones in front of car
            or located cones are sorted in natural way -> according to distance from car

            -> in one frame path prediction scenario we set car's coordinates as [0,0],
            thus sorting cones x axis seems reasonable

            TODO: invent something more robust
            """
            if Y.size != 0:
                Y = Y[np.argsort(Y[:, self.movement_direction])]
            else:
                B = B[np.argsort(B[:, self.movement_direction])]

            B, Y = self.fill_missing(B, Y)
        self.tmp = {"B": B, "Y": Y}

        if n_steps < 1:
            raise ValueError("Number of steps must be positive!!")

        # initializing loop
        # step 1)
        b_0 = self.find_closest_one(B)
        y_0 = self.find_closest_one(Y)
        self.sorted_blue_cones.append(b_0)
        self.sorted_yellow_cones.append(y_0)

        s_1 = self.calculate_center(b_0, y_0)
        self.start_points.append(s_1)

        # step 2)
        if n_steps > 1:
            try:
                # special case of separate line for 2nd step
                self.find_line_parameters(b_0, y_0, normal=False)
                B_hat = self.points_above_normal(B)
                Y_hat = self.points_above_normal(Y)
                b_1 = self.find_closest_one(B_hat)
                y_1 = self.find_closest_one(Y_hat)
                self.sorted_blue_cones.append(b_1)
                self.sorted_yellow_cones.append(y_1)

                s_2 = self.calculate_center(b_1, y_1)
                self.start_points.append(s_2)
            except ValueError as err:
                # catching specific error
                if str(err) == "attempt to get argmin of an empty sequence":
                    # warnings.warn("Too many iteration and not enough cones!")
                    pass
                else:
                    raise ValueError(str(err))
        # every other step
        if n_steps > 2:
            for step in range(n_steps - 2):
                try:
                    self.find_next_center(B, Y, step + 2, verbose=verbose)
                except ValueError as err:
                    # catching specific error
                    if str(err) == "attempt to get argmin of an empty sequence":
                        #     warnings.warn("Too many iteration! (n_step > 2)")
                        break
                    else:
                        raise ValueError(str(err))

    def fill_missing(self, B, Y):
        # set_trace()
        yellow = True if len(Y) > len(B) else False
        if yellow:
            to_fill = B
            full = Y
        else:
            to_fill = Y
            full = B
        if len(to_fill) == 0:
            to_fill = to_fill.reshape(-1, full.shape[1])

        for idx in range(len(to_fill), len(full)):
            if idx == 0:
                vec = full[1, :] - full[0, :]
                vec = vec / np.linalg.norm(vec)
                if yellow:
                    vec = np.matmul(np.array([[0, -1], [1, 0]]), vec)
                else:
                    vec = np.matmul(np.array([[0, 1], [-1, 0]]), vec)
            elif idx == len(full) - 1:
                vec = full[-1, :] - full[-2, :]
                vec = vec / np.linalg.norm(vec)
                if yellow:
                    vec = np.matmul(np.array([[0, -1], [1, 0]]), vec)
                else:
                    vec = np.matmul(np.array([[0, 1], [-1, 0]]), vec)
            else:
                vec1 = full[idx - 1, :] - full[idx, :]
                vec2 = full[idx + 1, :] - full[idx, :]
                vec1 = vec1 / (np.linalg.norm(vec1) + 1e-6)
                vec2 = vec2 / (np.linalg.norm(vec2) + 1e-6)
                angle = np.arccos(np.dot(vec1, vec2)) / 2
                if yellow:
                    vec = np.matmul(np.array(
                        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), vec2)
                else:
                    vec = np.matmul(
                        np.array([[np.cos(-angle), -np.sin(-angle)],
                                 [np.sin(-angle), np.cos(-angle)]]), vec2
                    )

            to_fill = np.vstack(
                (to_fill, full[idx, :] + vec * self.filling_cones_distance))
        # print(to_fill, full)
        if yellow:
            return to_fill, full
        else:
            return full, to_fill


class PathPlanner():
    def __init__(self, opt={"n_steps": 20}):
        self.n_steps = opt["n_steps"]
        self.planner = PathPlanning(np.array([0, 0]))

    def find_path(self, cones):
        self.planner.reset(np.array([0, 0]))
        if cones is None:
            blue_cones = np.zeros((0, 3))
            yellow_cones = np.zeros((0, 3))
        else:
            yellow_cones = cones[cones[:, 2] == ConeClasses.YELLOW, :]
            blue_cones = cones[cones[:, 2] == ConeClasses.BLUE, :]
        try:
            self.planner.find_path(blue_cones[:, :2], yellow_cones[:, :2], n_steps=self.n_steps)
            path = np.vstack(self.planner.start_points)
        except:
            path = np.array([[0., 0.]])

        path_sorted_idxs = path[:, 0].argsort()
        path = path[path_sorted_idxs]

        return path
