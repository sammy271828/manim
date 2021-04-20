import numpy as np
from scipy import optimize
from queue import Queue
from numpy import heaviside

import moderngl

from manimlib.constants import *
from manimlib.mobject.mobject import Mobject
from manimlib.utils.bezier import integer_interpolate
from manimlib.utils.bezier import interpolate
from manimlib.utils.images import get_full_raster_image_path
from manimlib.utils.iterables import listify
from manimlib.utils.space_ops import normalize_along_axis


class MySurface(Mobject):
    CONFIG = {
        # Resolution counts number of points sampled, which for
        # each coordinate is one more than the the number of
        # rows/columns of approximating squares
        "resolution": (101, 101),
        "color": RED,
        "opacity": 1.0,
        "gloss": 0.3,
        "shadow": 0.4,
        "prefered_creation_axis": 1,
        # For du and dv steps.  Much smaller and numerical error
        # can crop up in the shaders.
        "epsilon": 1e-5,
        "render_primitive": moderngl.TRIANGLES,
        "depth_test": True,
        "shader_folder": "surface",
        "shader_dtype": [
            ('point', np.float32, (3,)),
            ('du_point', np.float32, (3,)),
            ('dv_point', np.float32, (3,)),
            ('color', np.float32, (4,)),
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(len(self.get_triangle_indices()), len(self.get_points()))

    def init_points(self):
        func = lambda x, y, z: 81 * (x ** 3 + y ** 3 + z ** 3) - 189 * (x ** 2 * y + x ** 2 * z + x * y ** 2 + x * z ** 2 + y ** 2 * z + y * z ** 2) + 54 * x * y * z + 126 * (x * y + x * z + y * z) - 9 * (x ** 2 + y ** 2 + z ** 2) - 9 * (x + y + z) + 1
        partial_x = lambda x, y, z: 81 * (3 * x ** 2) - 189 * (2 * x * y + 2 * x * z + y ** 2 + z ** 2) + 54 * y * z + 126 * (y + z) - 9 * (2 * x) - 9
        partial_y = lambda x, y, z: partial_x(y, x, z)
        partial_z = lambda x, y, z: partial_x(z, x, y)

        a = 1 / 7

        epsilon = 1e-5
        print(epsilon)

        afunc = lambda x, y, z: func(a * x, a * y, a * z)

        points = []
        tri_indices = []
        point_list = []

        u_list = []
        v_list = []

        x_max = 7
        y_max = x_max
        z_max = 5

        resolution = 6

        ranges = np.array([x_max, y_max, z_max])

        ranges *= resolution
        std_basis = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])

        nudge = std_basis / resolution

        point_grid = np.zeros(shape=(2 * ranges[0] + 1, 2 * ranges[1] + 1, 2 * ranges[2] + 1))
        point_table = {}
        point_indices = {}

        flag = True
        first_point = [0, 0, 0]

        v1 = np.array([1,0,0])
        print(v1, "hi")

        # RETRIEVE POINTS OF MESH
        for i in range(-ranges[0], ranges[0]):
            for j in range(-ranges[1], ranges[1]):
                for k in range(-ranges[2], ranges[2]):

                    # if i**2 + j**2 + (2*k)**2 > 200:
                    #    continue

                    P = np.array([i / resolution, j / resolution, k / resolution])

                    if np.linalg.norm(P) > 6:
                        continue

                    Q = P + nudge[0]
                    R = P + nudge[1]
                    S = P + nudge[2]

                    fp = afunc(P[0], P[1], P[2])
                    fq = afunc(Q[0], Q[1], Q[2])
                    fr = afunc(R[0], R[1], R[2])
                    fs = afunc(S[0], S[1], S[2])

                    if fp * fq <= 0 or fp * fr <= 0 or fp * fs <= 0:
                        point_grid[ranges[0] + i][ranges[1] + j][ranges[2] + k] = 1
                        # point_table[(i,j,k)] = P

                        if fp * fq <= 0:
                            fun = lambda x: afunc(x, P[1], P[2])

                            X = optimize.newton(fun, P[0])

                            T = np.array([X, P[1], P[2]])
                            # dots.add(SmallDot(T))

                        elif fp * fr <= 0:
                            fun = lambda y: afunc(P[0], y, P[2])

                            Y = optimize.newton(fun, P[1])

                            T = np.array([P[0], Y, P[2]])
                            # dots.add(SmallDot(T))

                        elif fp * fs <= 0:
                            fun = lambda z: afunc(P[0], P[1], z)

                            Z = optimize.newton(fun, P[2])

                            T = np.array([P[0], P[1], Z])
                            # dots.add(SmallDot(T))

                        point_table[(i, j, k)] = T

                        point_indices[(i, j, k)] = len(points)
                        points.append((i, j, k))

                        point_list.append(T)

                        gradient = np.array([partial_x(*T), partial_y(*T), partial_z(*T)])


                        u_vec = np.cross(gradient, v1)
                        v_vec = np.cross(gradient, u_vec)

                        u_vec = u_vec / np.linalg.norm(u_vec)
                        v_vec = v_vec / np.linalg.norm(v_vec)

                        u_vec *= epsilon
                        v_vec *= epsilon

                        u_list.append(T + u_vec)
                        v_list.append(T + v_vec)

                        if flag:
                            flag = False
                            first_point = [i, j, k]

        # print(len(point_table))

        # TODO: FIX THIS!!
        # dummy search
        if False:
            for i in range(-ranges[0], ranges[0]):
                for j in range(-ranges[1], ranges[1]):
                    for k in range(-ranges[2], ranges[2]):
                        if point_grid[ranges[0] + i][ranges[1] + j][ranges[2] + k] == 1:
                            x = ranges[0] + i
                            y = ranges[1] + j
                            z = ranges[2] + k

                            coords = [x, y, z]

                            print(coords)

                            start_point = point_table[(i, j, k)]

                            for a in range(0, 3):
                                if point_grid[x + delta(a, 0)][y + delta(a, 1)][z + delta(a, 2)] == 1:
                                    end_point = point_table[(i + delta(a, 0), j + delta(a, 1), k + delta(a, 2))]
                                    pass

        # BREADTH-FIRST SEARCH
        if True:
            print(first_point)

            point_grid = np.pad(point_grid, (1, 0), mode='constant')

            x0 = ranges[0] + first_point[0] + 1
            y0 = ranges[1] + first_point[1] + 1
            z0 = ranges[2] + first_point[2] + 1

            print(point_grid[x0, y0, z0])

            print([x0, y0, z0])

            visited = {}
            num_points = len(point_table)

            coords = [x0, y0, z0]

            step = [-1, 0, 1]

            steps = []
            for a in range(-1, 2):
                for b in range(-1, 2):
                    for c in range(-1, 2):
                        steps.append(a * std_basis[0] + b * std_basis[1] + c * std_basis[2])

            for s in steps:
                print(s)

            next_points = Queue()

            next_points.put(first_point)

            triangles = 0

            while not next_points.empty():
                current = next_points.get()

                start_point = point_table[(current[0], current[1], current[2])]

                x0 = ranges[0] + current[0] + 1
                y0 = ranges[1] + current[1] + 1
                z0 = ranges[2] + current[2] + 1

                if point_grid[x0, y0, z0] == 0:
                    continue

                point_grid[x0, y0, z0] = 0

                box = np.zeros((3, 3, 3))

                if True:
                    for h in step:
                        for k in step:
                            for l in step:
                                x = x0 + h
                                y = y0 + k
                                z = z0 + l

                                if point_grid[x, y, z] == 1:
                                    box[h, k, l] = 1
                                    end_point = point_table[(x - ranges[0] - 1, y - ranges[1] - 1, z - ranges[2] - 1)]
                                    next_points.put([x - ranges[0] - 1, y - ranges[1] - 1, z - ranges[2] - 1])

                if True:
                    p0 = start_point
                    t0 = point_indices[(current[0], current[1], current[2])]
                    neighbours = []

                    for h in step:
                        for k in step:
                            for l in step:
                                x1 = x0 + h
                                y1 = y0 + k
                                z1 = z0 + l

                                if point_grid[x1, y1, z1] == 1:
                                    neighbours.append(np.array([x1, y1, z1]))
                                    box[h, k, l] = 1

                                    p1 = point_table[(x1 - ranges[0] - 1, y1 - ranges[1] - 1, z1 - ranges[2] - 1)]
                                    t1 = point_indices[(x1 - ranges[0] - 1, y1 - ranges[1] - 1, z1 - ranges[2] - 1)]

                                    next_points.put([x1 - ranges[0] - 1, y1 - ranges[1] - 1, z1 - ranges[2] - 1])

                                    for h in step:
                                        for k in step:
                                            for l in step:
                                                x2 = x1 + h
                                                y2 = y1 + k
                                                z2 = z1 + l

                                                if np.linalg.norm(np.array([x2, y2, z2]) - np.array([x0, y0, z0])) < 2:
                                                    if point_grid[x2, y2, z2] == 1:
                                                        p2 = point_table[(
                                                        x2 - ranges[0] - 1, y2 - ranges[1] - 1, z2 - ranges[2] - 1)]
                                                        t2 = point_indices[(
                                                        x2 - ranges[0] - 1, y2 - ranges[1] - 1, z2 - ranges[2] - 1)]

                                                        triangles += 1

                                                        tri_indices.append(t0)
                                                        tri_indices.append(t1)
                                                        tri_indices.append(t2)

                                                        # dots.add(Triangle3D(P0=p0,P1=p1,P2=p2, color = RED, resolution = (12,12)))

            pass

        tri_indices = np.asarray(tri_indices)


        point_lists = point_list + u_list + v_list

        point_lists = np.asarray(point_lists)

        self.set_points(point_lists)
        self.triangle_indices = tri_indices

    def get_surface_points_and_nudged_points(self):
        points = self.get_points()
        k = len(points) // 3
        return points[:k], points[k:2 * k], points[2 * k:]

    def get_triangle_indices(self):
        return self.triangle_indices

    def get_surface_points_and_nudged_points(self):
        points = self.get_points()
        k = len(points) // 3
        return points[:k], points[k:2 * k], points[2 * k:]

    # For shaders
    def get_shader_data(self):
        s_points, du_points, dv_points = self.get_surface_points_and_nudged_points()
        shader_data = self.get_resized_shader_data_array(len(s_points))
        if "points" not in self.locked_data_keys:
            shader_data["point"] = s_points
            shader_data["du_point"] = du_points
            shader_data["dv_point"] = dv_points
        self.fill_in_shader_color_info(shader_data)
        return shader_data

    def fill_in_shader_color_info(self, shader_data):
        self.read_data_to_shader(shader_data, "color", "rgbas")
        return shader_data

    def get_shader_vert_indices(self):
        return self.get_triangle_indices()
        
