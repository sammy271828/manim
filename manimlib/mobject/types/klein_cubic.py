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
from manimlib.utils.space_ops import normalize_along_axis, rotation_matrix

class Node:
    def __init__(self, data, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

    def print(self):
        print(self.data)

    def get_data(self):
        x=self.data
        return x

    def set_data(self, val):
        self.data = val


class KleinCubic(Mobject):
    CONFIG = {
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

    def init_points(self):

        #SCALING PARAMETER
        a = 3/14

        # STEP SIZE
        resolution = 0.15

        #NUMBER OF STEPS TAKEN
        steps = 2700
        epsilon = self.epsilon

        #DEFINING FUNCTION FOR THE KLEIN CUBIC
        a_3=81
        a_21 = -189

        afunc = lambda x, y, z: a_3 * (x ** 3 + y ** 3 + z ** 3) + a_21 * (x ** 2 * y + x ** 2 * z + x * y ** 2 + x * z ** 2 + y ** 2 * z + y * z ** 2) + 54 * x * y * z + 126 * (x * y + x * z + y * z) - 9 * (x ** 2 + y ** 2 + z ** 2) - 9 * (x + y + z) + 1
        apartial_x = lambda x, y, z: a_3 * (3 * x ** 2) + a_21 * (2 * x * y + 2 * x * z + y ** 2 + z ** 2) + 54 * y * z + 126 * (y + z) - 9 * (2 * x) - 9
        apartial_y = lambda x, y, z: apartial_x(y, x, z)
        apartial_z = lambda x, y, z: apartial_x(z, x, y)

        #SCALED VERSION
        func = lambda x, y, z: afunc(a * x, a * y, a * z)
        partial_x = lambda x, y, z: a * apartial_x(a * x, a * y, a * z)
        partial_y = lambda x, y, z: a * apartial_y(a * x, a * y, a * z)
        partial_z = lambda x, y, z: a * apartial_z(a * x, a * y, a * z)

        #DEFINING FUNCTION FOR THE SPHERE
        sphere = lambda x,y,z: x**2 + y**2 + z**2 - 36
        sphere_dx = lambda x,y,z: 2*x
        sphere_dy = lambda x,y,z: sphere_dx(y,x,z)
        sphere_dz = lambda x,y,z: sphere_dx(z,x,y)

        #AUXILIARY FUNCTION TO ENFORCE VANISHING OF BOTH CUBIC AND SPHERE EXPRESSIONS
        f = lambda x,y,z: (func(x,y,z))**2 + (sphere(x,y,z))**2
        f_x = lambda x,y,z: 2*(func(x,y,z))*partial_x(x,y,z) + 2*sphere(x,y,z)*sphere_dx(x,y,z)
        f_y = lambda x,y,z: f_x(y,x,z)
        f_z = lambda x,y,z: f_x(z,x,y)

        #FUNCTION FOR OPTIMIZING A GIVEN INITIAL GUESS USING NEWTON'S METHOD
        def get_point_on_surface(next, func = func, partials = [partial_x,partial_y,partial_z]):

            temp = ORIGIN

            while np.linalg.norm(next-temp) > 0.000001:
                temp=next
                normal = np.array([partials[0](*temp), partials[1](*temp), partials[2](*temp)])
                normal_length = np.linalg.norm(normal)

                #UPDATE
                next = temp - func(*temp)/(normal_length**2) * normal

            return next

        # FINDS ANGLE BETWEEN NEIGHBORING VECTORS AFTER PROJECTING TO TANGENT PLANE
        def angle(current):
            prev = current.prev.get_data()
            next = current.next.get_data()
            curr = current.get_data()

            vectors = [prev - curr, next - curr]

            # PROJECT VECTORS ONTO THE TANGENT PLANE AT THE CURRENT POINT
            normal = np.array([partial_x(*curr), partial_y(*curr), partial_z(*curr)])
            normal /= np.linalg.norm(normal)

            vectors[0] = vectors[0] - np.dot(vectors[0], normal) * normal
            vectors[1] = vectors[1] - np.dot(vectors[1], normal) * normal

            vectors[0] /= np.linalg.norm(vectors[0])
            vectors[1] /= np.linalg.norm(vectors[1])

            return np.dot(vectors[0], vectors[1])


        #SPLIT OUTER BOUNDARY POLYGON INTO THREE PARTS TO BE COMPUTED SEPARATELY
        plane_curve_one = []
        plane_curve_two = []
        boundary_curve = []

        # TO HOLD THE THREE BOUNDARY CURVES
        bd_polygon = []

        #INNER BOUNDARY POLYGON
        inner_curve = []

        # FOR ORGANIZING THE POINTS AND TRIANGLES
        point_list = []
        u_list = []
        v_list = []
        tri_indices = []
        index_table = {}
        
        
        #CALCULATING THE TWO CUTTING PLANES
        z_dir = np.array([1, 1, 1])
        x_dir = np.array([-1, -1, 2])
        y_dir = np.cross(z_dir, x_dir)

        z_dir = z_dir / np.linalg.norm(z_dir)
        x_dir = x_dir / np.linalg.norm(x_dir)
        y_dir = y_dir / np.linalg.norm(y_dir)

        m = rotation_matrix(angle=-PI / 3, axis=z_dir)
        purp_dir = np.matmul(m, x_dir)
        purp_dir /= np.linalg.norm(purp_dir)

        plane_normal = np.cross(z_dir, purp_dir)
        plane_normal /= np.linalg.norm(plane_normal)

        plane_normal_two = np.array([1,-1,0])
        plane_normal_two = plane_normal_two/np.linalg.norm(plane_normal_two)


        #FIND FIRST POINT ON BOUNDARY POLYGON
        point = np.array([2, 2, 2])
        newton_func = lambda t: func(*(point + t * point))
        T = optimize.newton(newton_func, 0)
        start = point + T * point

        #TRAVERSE FIRST PLANE CURVE UNTIL OUTER SPHERE IS REACHED
        current = start

        while np.linalg.norm(current) < 6:

            plane_curve_one.append(current)

            #COMPUTE NORMAL AND TANGENT VECTORS
            normal = np.array([partial_x(*current), partial_y(*current), partial_z(*current)])
            normal /= np.linalg.norm(normal)

            tangent = np.cross(normal,plane_normal)
            tangent /= np.linalg.norm(tangent)
            tangent *= resolution

            #VISUALIZING THE MARCHING ALGORITHM WITH NORMAL AND TANGENT VECTORS
            begin = current

            #CALCULATE NEXT STEP
            current = current + tangent

            next=current
            temp=ORIGIN

            #CALCULATE POINT ON SURFACE USING NEWTON'S METHOD
            current = get_point_on_surface(next)


        # FIRST POINT ON BOUNDARY CURVE LYING ON SPHERE VIA NEWTON'S METHOD
        next=current
        current = get_point_on_surface(next, func=f, partials = [f_x,f_y,f_z])

        #MARCH ALONG BOUNDARY CURVE
        while np.dot(plane_normal_two,current) < 0:
            boundary_curve.append(current)

            # COMPUTE NORMAL AND TANGENT VECTORS
            normal = np.array([partial_x(*current), partial_y(*current), partial_z(*current)])
            normal /= np.linalg.norm(normal)

            sphere_normal = np.array([sphere_dx(*current), sphere_dy(*current), sphere_dz(*current)])
            sphere_normal /= np.linalg.norm(sphere_normal)

            tangent = np.cross(sphere_normal,normal)
            tangent /= np.linalg.norm(tangent)
            tangent *= resolution

            # CALCULATE NEXT STEP
            current = current + tangent
            next = current
            current = get_point_on_surface(next, func=f, partials = [f_x,f_y,f_z])


        #ORTHOGONAL PROJECTION ONTO THE SECOND PLANE
        current = np.dot(current,x_dir) * x_dir + np.dot(current,z_dir) * z_dir

        #FIRST POINT ON SECOND PLANE CURVE VIA NEWTON'S METHOD
        next = current
        current = get_point_on_surface(next, func=f, partials=[f_x, f_y, f_z])

        # MARCH ALONG SECOND PLANE CURVE
        while np.dot(x_dir,current) > 0:
            plane_curve_two.append(current)

            # COMPUTE NORMAL AND TANGENT VECTORS
            normal = np.array([partial_x(*current), partial_y(*current), partial_z(*current)])
            normal /= np.linalg.norm(normal)

            tangent = np.cross(plane_normal_two,normal)
            tangent /= np.linalg.norm(tangent)
            tangent *= resolution

            # CALCULATE NEXT STEP
            current = current + tangent
            next = current
            current = get_point_on_surface(next)


    #FIRST POINT ON INNER CURVE
        #TODO: make this not hard-coded
        first = np.array([0.16105883, 0.04729694, 0.66286412])

        #PROJECT ONTO THE PLANE
        print(np.dot(first, plane_normal_two))
        first = first - np.dot(first,plane_normal_two)*plane_normal_two

        #FIND FIRST POINT ON THE CURVE
        next = first
        current = get_point_on_surface(next)
        first = current

        while len(inner_curve) < 3 or np.linalg.norm(current - first) > resolution:
            inner_curve.append(current)

            # COMPUTE NORMAL AND TANGENT VECTORS
            normal = np.array([partial_x(*current), partial_y(*current), partial_z(*current)])
            normal /= np.linalg.norm(normal)

            tangent = np.cross(plane_normal_two, normal)
            tangent /= np.linalg.norm(tangent)
            tangent *= resolution

            # CALCULATE NEXT STEP
            current = current + tangent

            next = current
            current = get_point_on_surface(next)


    #CALCULATE TRIANGLES
        bd_polygon = plane_curve_one + boundary_curve + plane_curve_two

        # PUT INNER POLYGON IN CLOCKWISE CIRCULAR LIST TO BE JOINED LATER
        inner_size = len(inner_curve)

        inner_head = Node(inner_curve[0])
        current = Node(inner_curve[1])

        inner_head.next = current
        current.prev = inner_head

        for i in range(2, inner_size):
            next = Node(inner_curve[i])
            next.prev = current
            current.next = next
            current = next

        current.next = inner_head
        inner_head.prev = current

        #PUT OUTER POLYGON IN CIRCULAR LIST TO MAKE UPDATING EASIER
        head = Node(bd_polygon[0])
        current = Node(bd_polygon[1])

        head.next = current
        current.prev = head

        for i in range(2,len(bd_polygon)):
            next = Node(bd_polygon[i])
            next.prev = current
            current.next = next
            current = next

        current.next = head
        head.prev = current

        # COMPUTE MESH
        current = head
        inner_reached = False

        for i in range(steps):
            #CHECK IF CLOSE TO INNER BOUNDARY CURVE AND JOIN POLYGONS TOGETHER
            if not inner_reached and np.linalg.norm(current.get_data() - inner_head.get_data()) < resolution:

                # COMPUTE TRIANGLES
                new_tri = [current.get_data(), inner_head.get_data(), inner_head.prev.get_data()]
                for x in new_tri:
                    if (x[0], x[1], x[2]) not in index_table:
                        point_list.append(x)
                        index_table[(x[0], x[1], x[2])] = len(point_list) - 1
                    tri_indices.append(index_table[(x[0], x[1], x[2])])

                new_tri = [current.get_data(), current.next.get_data(),inner_head.prev.get_data()]
                for x in new_tri:
                    if (x[0], x[1], x[2]) not in index_table:
                        point_list.append(x)
                        index_table[(x[0], x[1], x[2])] = len(point_list) - 1
                    tri_indices.append(index_table[(x[0], x[1], x[2])])

                temp = current.next
                inner_temp = inner_head.prev

                current.next = inner_head
                inner_head.prev = current

                inner_temp.next = temp
                temp.prev = inner_temp

                inner_reached = True

            prev = current.prev.get_data()
            next = current.next.get_data()
            curr = current.get_data()

            vectors = [prev-curr, next-curr]

            #PROJECT VECTORS ONTO THE TANGENT PLANE AT THE CURRENT POINT
            normal = np.array([partial_x(*curr),partial_y(*curr),partial_z(*curr)])
            normal /= np.linalg.norm(normal)

            vectors[0] = vectors[0] - np.dot(vectors[0],normal) * normal
            vectors[1] = vectors[1] - np.dot(vectors[1],normal) * normal

            vectors[0] /= np.linalg.norm(vectors[0])
            vectors[1] /= np.linalg.norm(vectors[1])

            #IF ANGLE BETWEEN NEIGHBORING DIRECTIONS IS SMALL ENOUGH, JOIN THE POINTS
            if angle(current) > 0:
                current.prev.next = current.next
                current.next.prev = current.prev

                #COMPUTE TRIANGLES
                new_tri = [current.prev.get_data(),current.next.get_data(),current.get_data()]

                for x in new_tri:
                    if (x[0],x[1],x[2]) not in index_table:
                        point_list.append(x)
                        index_table[(x[0],x[1],x[2])] = len(point_list) - 1

                    tri_indices.append(index_table[(x[0],x[1],x[2])])

                current = current.next

            elif angle(current.next) > 0:
                new_tri = [current.next.next.get_data(), current.next.get_data(), current.get_data()]

                current.next = current.next.next
                current.next.prev = current

                # COMPUTE TRIANGLES
                for x in new_tri:
                    if (x[0], x[1], x[2]) not in index_table:
                        point_list.append(x)
                        index_table[(x[0], x[1], x[2])] = len(point_list) - 1

                    tri_indices.append(index_table[(x[0], x[1], x[2])])

            #CALCULATE THE UPDATED CURRENT POINT
            else:
                m = rotation_matrix(angle=-PI / 3, axis=normal)
                step = np.matmul(m,vectors[1])
                step *= resolution

                if np.dot(curr+step,plane_normal_two) > 0:
                    m = rotation_matrix(angle=2*PI / 3, axis=normal)
                    step = np.matmul(m,step)

                curr = curr + step

                point = curr
                point = get_point_on_surface(point)
                new_point = Node(data=point)

                #UPDATE OUTER POLYGON
                new_point.prev=current
                current=current.next
                new_point.prev.next=new_point
                current.prev=new_point
                new_point.next=current

                #TRIANGLES
                new_tri = [new_point.prev.get_data(),new_point.get_data(),current.get_data()]
                for x in new_tri:
                    if (x[0], x[1], x[2]) not in index_table:
                        point_list.append(x)
                        index_table[(x[0], x[1], x[2])] = len(point_list) - 1

                    tri_indices.append(index_table[(x[0], x[1], x[2])])

                new_tri = [new_point.prev.get_data(), new_point.prev.prev.get_data(), new_point.get_data()]
                for x in new_tri:
                    if (x[0], x[1], x[2]) not in index_table:
                        point_list.append(x)
                        index_table[(x[0], x[1], x[2])] = len(point_list) - 1

                    tri_indices.append(index_table[(x[0], x[1], x[2])])

                new_point.prev = new_point.prev.prev
                new_point.prev.next = new_point


        #COMPUTE U- AND V-LISTS
        for p in point_list:
            normal = (partial_x(*p), partial_y(*p), partial_z(*p))

            u_vec = np.cross(normal,plane_normal_two)
            v_vec = np.cross(normal,u_vec)

            u_vec = epsilon*u_vec/np.linalg.norm(u_vec)
            v_vec = epsilon*v_vec/np.linalg.norm(v_vec)

            u_list.append(p+u_vec)
            v_list.append(p+v_vec)

        # GET REFLECTED POINTS
        new_point_list = []
        new_u_list = []
        new_v_list = []
        new_tri_indices = []

        N = len(point_list)
        for i in range(0, N):
            p = point_list[i]
            u = u_list[i]
            v = v_list[i]

            comp = np.dot(p, plane_normal_two)
            new_p = p - 2 * comp * plane_normal_two

            comp = np.dot(u, plane_normal_two)
            new_u = u - 2 * comp * plane_normal_two

            comp = np.dot(v, plane_normal_two)
            new_v = v - 2 * comp * plane_normal_two

            point_list.append(new_p)
            u_list.append(new_u)
            v_list.append(new_v)

        for j in tri_indices:
            new_tri_indices.append(j + N)

        tri_indices = tri_indices + new_tri_indices

        # GET ROTATED POINTS
        new_tri_indices = []
        N = len(point_list)
        m = rotation_matrix(axis=z_dir, angle=TAU / 3)
        m2 = rotation_matrix(axis=z_dir, angle=2*TAU / 3)

        for i in range(0, N):
            p = point_list[i]
            u = u_list[i]
            v = v_list[i]

            new_p = np.matmul(m, p)
            new_u = np.matmul(m, u)
            new_v = np.matmul(m, v)

            point_list.append(new_p)
            u_list.append(new_u)
            v_list.append(new_v)

        for j in tri_indices:
            new_tri_indices.append(j + N)

        for i in range(0, N):
            p = point_list[i]
            u = u_list[i]
            v = v_list[i]

            new_p = np.matmul(m2, p)
            new_u = np.matmul(m2, u)
            new_v = np.matmul(m2, v)

            point_list.append(new_p)
            u_list.append(new_u)
            v_list.append(new_v)

        for j in tri_indices:
            new_tri_indices.append(j + 2 * N)

        tri_indices = tri_indices + new_tri_indices

        # PUTTING IT ALL TOGETHER
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
        
