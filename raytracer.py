import numpy as np
import traceback
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import os

# Try to import CuPy, but don't raise an error if it's not installed
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    use_gpu=True
except (ImportError, ModuleNameError, NameError) as e:
    CUPY_AVAILABLE = False
    use_gpu=False
    print("CuPy not found. GPU acceleration will not be available.")

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Material:
    def __init__(self, color, specular=0, reflection=0, refraction=0, refractive_index=1, metallic=0):
        self.color = color
        self.specular = specular
        self.reflection = reflection
        self.refraction = refraction
        self.refractive_index = refractive_index
        self.metallic = metallic

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

class Cylinder:
    def __init__(self, center, radius, height, material):
        self.center = center
        self.radius = radius
        self.height = height
        self.material = material

class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal = normal
        self.material = material

class Rectangle:
    def __init__(self, corner, u, v, material):
        self.corner = corner
        self.u = u
        self.v = v
        self.material = material

class Cube:
    def __init__(self, min_point, max_point, material):
        self.min_point = min_point
        self.max_point = max_point
        self.material = material

def normalize(v):
    return v / np.linalg.norm(v)

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def refract(incident, normal, n1, n2):
    # Simplified refraction calculation
    ratio = n1 / n2
    cos_i = -np.dot(normal, incident)
    sin_t_sq = ratio * ratio * (1.0 - cos_i * cos_i)
    if sin_t_sq > 1.0:
        return None
    cos_t = np.sqrt(1.0 - sin_t_sq)
    return ratio * incident + (ratio * cos_i - cos_t) * normal

def is_in_shadow(hit_point, light_pos, scene_data):
    light_dir = normalize(light_pos - hit_point)
    light_distance = np.linalg.norm(light_pos - hit_point)

    for obj_type in ['spheres', 'cylinders', 'planes', 'rectangles', 'cubes']:
        for obj in scene_data[obj_type]:
            t = globals()[f"intersect_{obj_type[:-1]}"](hit_point, light_dir, obj)
            if 0 < t < light_distance:
                return True
    return False


def intersect_sphere(ray_origin, ray_direction, sphere):
    center = sphere[:3]
    radius = sphere[3]
    oc = ray_origin - center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return np.inf
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if t > 0:
        return t
    return np.inf

def intersect_cylinder(ray_origin, ray_direction, cylinder):
    center = cylinder[:3]
    radius = cylinder[3]
    height = cylinder[4]
    axis = np.array([0, 1, 0])  # Assuming cylinder is aligned with y-axis

    oc = ray_origin - center

    a = np.dot(ray_direction, ray_direction) - np.dot(ray_direction, axis)**2
    b = 2 * (np.dot(ray_direction, oc) - np.dot(ray_direction, axis) * np.dot(oc, axis))
    c = np.dot(oc, oc) - np.dot(oc, axis)**2 - radius**2

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return np.inf

    t = (-b - np.sqrt(discriminant)) / (2 * a)
    if t < 0:
        t = (-b + np.sqrt(discriminant)) / (2 * a)

    if t < 0:
        return np.inf

    hit_point = ray_origin + t * ray_direction
    hit_height = np.dot(hit_point - center, axis)

    if 0 <= hit_height <= height:
        return t

    return np.inf

def intersect_plane(ray_origin, ray_direction, plane):
    point = plane[:3]
    normal = plane[3:6]
    denom = np.dot(normal, ray_direction)
    if abs(denom) > 1e-6:
        t = np.dot(point - ray_origin, normal) / denom
        if t >= 0:
            return t
    return np.inf

def intersect_rectangle(ray_origin, ray_direction, rectangle):
    corner = rectangle[:3]
    u, v = rectangle[3:6], rectangle[6:9]
    normal = np.cross(u, v)
    normal = normal / np.linalg.norm(normal)

    t = np.dot(corner - ray_origin, normal) / np.dot(ray_direction, normal)
    if t < 0:
        return np.inf

    hit_point = ray_origin + t * ray_direction
    w = hit_point - corner

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv
    s = (uv * wv - vv * wu) / D
    if s < 0 or s > 1:
        return np.inf

    t = (uv * wu - uu * wv) / D
    if t < 0 or t > 1:
        return np.inf

    return np.dot(corner - ray_origin, normal) / np.dot(ray_direction, normal)

def intersect_cube(ray_origin, ray_direction, cube):
    min_point = cube[:3]
    max_point = cube[3:6]

    t_min = (min_point - ray_origin) / ray_direction
    t_max = (max_point - ray_origin) / ray_direction

    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)

    t_near = np.max(t1)
    t_far = np.min(t2)

    if t_near > t_far or t_far < 0:
        return np.inf

    return t_near if t_near > 0 else t_far

def get_sphere_properties(obj, hit_point):
    normal = normalize(hit_point - obj[:3])
    return normal, obj[4:7], obj[7], obj[8], obj[9], obj[10]

def get_cylinder_properties(obj, hit_point):
    center = obj[:3]
    axis = normalize(obj[5:8])
    hit_point_on_base = hit_point - axis * np.dot(hit_point - center, axis)
    normal = normalize(hit_point_on_base - center)
    return normal, obj[8:11], obj[11], obj[12], obj[13], obj[14]

def get_plane_properties(obj, hit_point):
    normal = obj[:3]
    x, z = hit_point[0], hit_point[2]
    color = obj[4:7] if (int(x) % 2 == int(z) % 2) else obj[4:7] * 0.6
    return normal, color, obj[7], obj[8], 0, 1

def get_rectangle_properties(obj, hit_point):
    normal = obj[3:6]
    return normal, obj[9:12], obj[12], obj[13], 0, 1

def get_cube_properties(obj, hit_point):
    min_point, max_point = obj[:3], obj[3:6]
    normal = np.zeros(3)
    for i in range(3):
        if abs(hit_point[i] - min_point[i]) < 1e-6:
            normal[i] = -1
        elif abs(hit_point[i] - max_point[i]) < 1e-6:
            normal[i] = 1
    normal = normalize(normal)
    return normal, obj[6:9], obj[9], obj[10], obj[11], obj[12]

obj_property_functions = {
    'sphere': get_sphere_properties,
    'cylinder': get_cylinder_properties,
    'plane': get_plane_properties,
    'rectangle': get_rectangle_properties,
    'cube': get_cube_properties
}

def initialize_data(scene, camera):
    data = {}

    data['spheres'] = np.array([
        [s.center.x, s.center.y, s.center.z, s.radius,
         s.material.color.x, s.material.color.y, s.material.color.z,
         s.material.specular, s.material.reflection, s.material.refraction, s.material.refractive_index, s.material.metallic]
        for s in scene['spheres']
    ], dtype=np.float32)

    data['cylinders'] = np.array([
        [c.center.x, c.center.y, c.center.z, c.radius, c.height,
         c.material.color.x, c.material.color.y, c.material.color.z,
         c.material.specular, c.material.reflection, c.material.refraction, c.material.refractive_index, c.material.metallic]
        for c in scene['cylinders']
    ], dtype=np.float32)

    data['planes'] = np.array([
        [p.point.x, p.point.y, p.point.z,
         p.normal.x, p.normal.y, p.normal.z,
         p.material.color.x, p.material.color.y, p.material.color.z,
         p.material.specular, p.material.reflection]
        for p in scene['planes']
    ], dtype=np.float32)

    data['rectangles'] = np.array([
        [r.corner.x, r.corner.y, r.corner.z,
         r.u.x, r.u.y, r.u.z,
         r.v.x, r.v.y, r.v.z,
         r.material.color.x, r.material.color.y, r.material.color.z,
         r.material.specular, r.material.reflection, r.material.refraction, r.material.refractive_index, r.material.metallic]
        for r in scene['rectangles']
    ], dtype=np.float32)

    data['cubes'] = np.array([
        [c.min_point.x, c.min_point.y, c.min_point.z,
         c.max_point.x, c.max_point.y, c.max_point.z,
         c.material.color.x, c.material.color.y, c.material.color.z,
         c.material.specular, c.material.reflection, c.material.refraction, c.material.refractive_index, c.material.metallic]
        for c in scene['cubes']
    ], dtype=np.float32)

    data['camera'] = np.array([
        camera['position'].x, camera['position'].y, camera['position'].z,
        camera['look_at'].x, camera['look_at'].y, camera['look_at'].z,
        camera['up'].x, camera['up'].y, camera['up'].z,
        camera['fov']
    ], dtype=np.float32)

    return data

def find_closest_intersection(ray_origin, ray_direction, scene_data):
    closest_t = np.inf
    closest_obj = None
    obj_type = None

    for obj_name, intersect_func in [('spheres', intersect_sphere),
                                     ('cylinders', intersect_cylinder),
                                     ('planes', intersect_plane),
                                     ('rectangles', intersect_rectangle),
                                     ('cubes', intersect_cube)]:
        for obj in scene_data[obj_name]:
            t = intersect_func(ray_origin, ray_direction, obj)
            if t < closest_t:
                closest_t = t
                closest_obj = obj
                obj_type = obj_name[:-1]

    return closest_t, closest_obj, obj_type

def is_in_shadow(hit_point, light_pos, scene_data):
    light_dir = normalize(light_pos - hit_point)
    light_distance = np.linalg.norm(light_pos - hit_point)

    for obj_name, intersect_func in [('spheres', intersect_sphere),
                                     ('cylinders', intersect_cylinder),
                                     ('planes', intersect_plane),
                                     ('rectangles', intersect_rectangle),
                                     ('cubes', intersect_cube)]:
        for obj in scene_data[obj_name]:
            t = intersect_func(hit_point + light_dir * 0.001, light_dir, obj)
            if 0 < t < light_distance:
                return True
    return False

def trace_ray(ray_origin, ray_direction, scene_data, depth=0):
    if depth > 5:
        return np.zeros(3)

    closest_t, closest_obj, obj_type = find_closest_intersection(ray_origin, ray_direction, scene_data)

    if closest_obj is None:
        return np.array([0.5, 0.7, 1.0]) * (0.5 * ray_direction[1] + 0.5)

    hit_point = ray_origin + closest_t * ray_direction
    normal, color, specular, reflection, refraction, refractive_index = obj_property_functions[obj_type](closest_obj, hit_point)

    if np.dot(normal, ray_direction) > 0:
        normal = -normal

    light_pos = np.array([5, 5, 0])
    light_dir = normalize(light_pos - hit_point)
    view_dir = normalize(ray_origin - hit_point)

    # Ambient
    ambient = color * 0.1

    # Diffuse
    diffuse = np.maximum(np.dot(normal, light_dir), 0) * color

    # Specular
    reflect_dir = reflect(-light_dir, normal)
    specular_intensity = np.power(np.maximum(np.dot(view_dir, reflect_dir), 0), 32) * specular

    # Shadows
    shadow_factor = 0.5 if is_in_shadow(hit_point + normal * 0.001, light_pos, scene_data) else 1.0

    # Combine lighting
    color = ambient + (diffuse + specular_intensity) * shadow_factor

    # Reflection
    if reflection > 0 and depth < 5:
        reflect_dir = reflect(ray_direction, normal)
        reflect_origin = hit_point + normal * 0.001
        reflect_color = trace_ray(reflect_origin, reflect_dir, scene_data, depth + 1)
        color = color * (1 - reflection) + reflect_color * reflection

    # Refraction
    if refraction > 0 and depth < 5:
        refract_dir = refract(ray_direction, normal, refractive_index)
        if refract_dir is not None:
            refract_origin = hit_point - normal * 0.001
            refract_color = trace_ray(refract_origin, refract_dir, scene_data, depth + 1)
            color = color * (1 - refraction) + refract_color * refraction

    return np.clip(color, 0, 1)

def cpu_ray_color(ray_origin, ray_direction, scene_data, depth=0):
    if depth > 5:
        return np.zeros(3)

    closest_t = np.inf
    closest_obj = None
    obj_type = None

    for obj_name in ['spheres', 'cylinders', 'planes', 'rectangles', 'cubes']:
        for obj in scene_data[obj_name]:
            t = globals()[f"intersect_{obj_name[:-1]}"](ray_origin, ray_direction, obj)
            if t < closest_t:
                closest_t = t
                closest_obj = obj
                obj_type = obj_name[:-1]

    if closest_obj is None:
        return np.array([0.5, 0.7, 1.0]) * (0.5 * ray_direction[1] + 0.5)

    hit_point = ray_origin + closest_t * ray_direction

    if obj_type == 'sphere':
        normal = normalize(hit_point - closest_obj[:3])
        color = closest_obj[4:7]
        specular = closest_obj[7]
        reflection = closest_obj[8]
        refraction = closest_obj[9] if len(closest_obj) > 9 else 0
        refractive_index = closest_obj[10] if len(closest_obj) > 10 else 1
        metallic = closest_obj[11] if len(closest_obj) > 11 else 0
    elif obj_type == 'cylinder':
        center = closest_obj[:3]
        radius = closest_obj[3]
        height = closest_obj[4]
        axis = np.array([0, 1, 0])  # Assuming cylinder is aligned with y-axis
        cp = hit_point - center
        cp[1] = 0
        normal = normalize(cp)
        color = closest_obj[5:8]
        specular, reflection, refraction, refractive_index, metallic = closest_obj[8:13]
    elif obj_type == 'plane':
        normal = closest_obj[3:6]
        if np.dot(normal, ray_direction) > 0:
            normal = -normal
        x, z = hit_point[0], hit_point[2]
        color = closest_obj[6:9] if ((int(x * 2) % 2) ^ (int(z * 2) % 2)) else closest_obj[6:9] * 0.8
        specular = closest_obj[9]
        reflection = closest_obj[10]
        refraction = 0
        refractive_index = 1
        metallic = 0
    elif obj_type == 'rectangle':
        u, v = closest_obj[3:6], closest_obj[6:9]
        normal = normalize(np.cross(u, v))
        if np.dot(normal, ray_direction) > 0:
            normal = -normal
        color = closest_obj[9:12]
        specular, reflection, refraction, refractive_index, metallic = closest_obj[12:17]
    else:  # cube
        min_point, max_point = closest_obj[:3], closest_obj[3:6]
        normal = np.zeros(3)
        for i in range(3):
            if abs(hit_point[i] - min_point[i]) < 1e-6:
                normal[i] = -1
            elif abs(hit_point[i] - max_point[i]) < 1e-6:
                normal[i] = 1
        normal = normalize(normal)
        color = closest_obj[6:9]
        specular, reflection, refraction, refractive_index, metallic = closest_obj[9:14]

    light_pos = np.array([5, 5, 5])
    light_dir = normalize(light_pos - hit_point)
    view_dir = normalize(ray_origin - hit_point)

    # Ambient
    ambient = color * 0.1

    # Diffuse
    diffuse = np.maximum(np.dot(normal, light_dir), 0) * color

    # Specular
    reflect_dir = reflect(-light_dir, normal)
    specular_intensity = np.power(np.maximum(np.dot(view_dir, reflect_dir), 0), 32) * specular

    # Shadow
    shadow_factor = 0.5 if is_in_shadow(hit_point + normal * 0.001, light_pos, scene_data) else 1.0

    # Metallic reflection
    if metallic > 0:
        metallic_reflection = cpu_ray_color(hit_point + normal * 0.001, reflect(-view_dir, normal), scene_data, depth + 1)
        color = color * (1 - metallic) + metallic_reflection * metallic

    local_color = ambient + (diffuse + specular_intensity) * shadow_factor

    # Reflection
    reflection_color = np.zeros(3)
    if reflection > 0 and depth < 5:
        reflect_dir = reflect(ray_direction, normal)
        reflect_origin = hit_point + normal * 0.001
        reflection_color = cpu_ray_color(reflect_origin, reflect_dir, scene_data, depth + 1)

    # Refraction
    refraction_color = np.zeros(3)
    if refraction > 0 and depth < 5:
        refract_dir = refract(ray_direction, normal, 1.0, refractive_index)
        if refract_dir is not None:
            refract_origin = hit_point - normal * 0.001
            refraction_color = cpu_ray_color(refract_origin, refract_dir, scene_data, depth + 1)

    # Combine colors
    color = local_color * (1 - reflection - refraction) + reflection_color * reflection + refraction_color * refraction

    return np.clip(color, 0, 1)



def render_tile(args):
    x_start, x_end, y_start, y_end, width, height, scene_data, samples, camera = args
    tile = np.zeros((y_end - y_start, x_end - x_start, 3))

    camera_position = camera['position']
    camera_look_at = camera['look_at']
    camera_up = camera['up']

    camera_forward = normalize(camera_look_at - camera_position)
    camera_right = normalize(np.cross(camera_forward, camera_up))
    camera_up = np.cross(camera_right, camera_forward)

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            color = np.zeros(3)
            for _ in range(samples):
                u = (x + np.random.random()) / width
                v = (y + np.random.random()) / height
                direction = normalize(
                    camera_right * (2 * u - 1) * width / height +
                    camera_up * (1 - 2 * v) +
                    camera_forward
                )
                color += cpu_ray_color(camera_position, direction, scene_data)
            tile[y - y_start, x - x_start] = color / samples
    return tile

def render_cpu(width, height, samples, data, camera):
    num_threads = cpu_count()
    chunk_height = height // num_threads

    pool = Pool(processes=num_threads)
    chunks = []
    for i in range(num_threads):
        y_start = i * chunk_height
        y_end = y_start + chunk_height if i < num_threads - 1 else height
        chunks.append((0, width, y_start, y_end, width, height, data, samples, camera))

    results = list(tqdm(pool.imap(render_tile, chunks), total=len(chunks), desc="Rendering"))

    return np.vstack(results)


# def render_gpu(width, height, samples, data, camera):
#     output = cp.zeros((height, width, 3), dtype=cp.float32)
#
#     # Convert data to CuPy arrays
#     gpu_data = {k: cp.array(v) for k, v in data.items()}
#
#     threadsperblock = (16, 16)
#     blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
#     blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
#     blockspergrid = (blockspergrid_x, blockspergrid_y)
#
#     ray_trace_kernel(
#         (blockspergrid_x, blockspergrid_y),
#         threadsperblock,
#         (output, width, height, samples,
#          gpu_data['spheres'], gpu_data['spheres'].shape[0],
#          gpu_data['cylinders'], gpu_data['cylinders'].shape[0],
#          gpu_data['planes'], gpu_data['planes'].shape[0],
#          gpu_data['rectangles'], gpu_data['rectangles'].shape[0],
#          gpu_data['cubes'], gpu_data['cubes'].shape[0],
#          gpu_data['camera'])
#     )
#
#     return output
def render_gpu(width, height, samples, gpu_data):
    output = cp.zeros((height, width, 3), dtype=cp.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]

    # Debug: Print shapes of input data
    # for key, value in gpu_data.items():
    #     if isinstance(value, cp.ndarray):
    #         print(f"{key} shape: {value.shape}, dtype: {value.dtype}")
    #     else:
    #         print(f"{key}: {value}")

    try:

        ray_trace_kernel(
            (blockspergrid_x, blockspergrid_y),
            threadsperblock,
            (output, cp.int32(width), cp.int32(height), cp.int32(samples),
             gpu_data['spheres'], cp.int32(gpu_data['num_spheres']),
             gpu_data['cylinders'], cp.int32(gpu_data['num_cylinders']),
             gpu_data['planes'], cp.int32(gpu_data['num_planes']),
             gpu_data['rectangles'], cp.int32(gpu_data['num_rectangles']),
             gpu_data['cubes'], cp.int32(gpu_data['num_cubes']),
             gpu_data['camera'], gpu_data['light'])
        )

        cp.cuda.stream.get_current_stream().synchronize()
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Error: {e}")
        return None


    # Debug: Check if output is all the same color
    # if cp.all(output == output[0, 0]):
    #     print("Warning: Output is a solid color")

    return output


def render(width, height, samples, scene, gpu_data, camera, use_gpu):
    if use_gpu and CUPY_AVAILABLE:
        try:
            print("Attempting GPU rendering...")
            image = render_gpu(width, height, samples, gpu_data)
            if image is None:
                print("GPU rendering failed, falling back to CPU")
                image = render_cpu(width, height, samples, scene, camera)
            else:
                print("GPU rendering successful.")
                # Convert CuPy array to NumPy array
                image = cp.asnumpy(image)
        except Exception as e:
            print(f"Error during GPU rendering: {e}")
            print(full_stack())
            print("Falling back to CPU rendering")
            image = render_cpu(width, height, samples, scene, camera)
    else:
        print("Using CPU rendering...")
        image = render_cpu(width, height, samples, scene, camera)

    # Ensure the image is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Ensure the image is in the correct format
    image = np.clip(image, 0, 1)  # Clip values between 0 and 1
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit format
    return image


def full_stack():
    import traceback, sys
    exc = sys.exc_info()[0]
    if exc is not None:
        f = sys.exc_info()[-1].tb_frame.f_back
        stack = traceback.extract_stack(f)
    else:
        stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
        stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr

# def main():
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Ray Tracing Renderer")
#     parser.add_argument("--cpu", action="store_true", help="Force CPU rendering")
#     parser.add_argument("--gpu", action="store_true", help="Force GPU rendering")
#     parser.add_argument("--samples", type=int, default=4, help="Number of samples per pixel")
#     parser.add_argument("--width", type=int, default=800, help="Image width")
#     parser.add_argument("--height", type=int, default=600, help="Image height")
#
#     args = parser.parse_args()
#
#     # Determine rendering mode
#     if args.cpu:
#         use_gpu = False
#     elif args.gpu:
#         use_gpu = True
#     else:
#         # Automatically choose based on CuPy availability
#         use_gpu = CUPY_AVAILABLE
#
#     width, height = args.width, args.height
#     samples = args.samples
#
#     print(f"Rendering at {width}x{height} with {samples} samples per pixel")
#     print(f"Using {'GPU' if use_gpu else 'CPU'} rendering")
#
#     # Camera setup
#     camera = {
#         'position': Vector3(-1, 1.5, 2),
#         'look_at': Vector3(0, 0, -5),
#         'up': Vector3(0, 1, 0),
#         'fov': 60.0  # Field of view in degrees
#     }
#
#
#     # Adjusted scene setup
#     scene = {
#         'spheres': [
#             Sphere(Vector3(0, 1, -5), 1, Material(Vector3(1, 0, 0), specular=0.6, reflection=0.2, refraction=0.8, refractive_index=1.5)),
#             Sphere(Vector3(-2.5, 1, -7), 1.5, Material(Vector3(0, 1, 0), specular=0.4, reflection=0.8, metallic=0.8)),
#             Sphere(Vector3(2.5, 1, -6), 0.75, Material(Vector3(0, 0, 1), specular=0.5, reflection=0.1, refraction=0.9, refractive_index=1.5))
#         ],
#         'cylinders': [
#             Cylinder(Vector3(-1, 0, -4), 0.5, 2, Material(Vector3(1, 0, 1), specular=0.7, reflection=0.1, refraction=0.8, refractive_index=1.5))
#         ],
#         'planes': [
#             Plane(Vector3(0, -1, 0), Vector3(0, 1, 0), Material(Vector3(0.5, 0.5, 0.5), specular=0.1, reflection=0.1)),
#             Plane(Vector3(-5, 0, 0), Vector3(1, 0, 0), Material(Vector3(0.5, 0.5, 0.5), specular=0.1, reflection=0.1)),
#             Plane(Vector3(0, 0, -10), Vector3(0, 0, 1), Material(Vector3(0.5, 0.5, 0.5), specular=0.1, reflection=0.1))
#         ],
#         'rectangles': [
#             Rectangle(Vector3(1, 0, -4), Vector3(0.5, 0, 0), Vector3(0, 1, 0), Material(Vector3(1, 0.5, 0), specular=0.3, reflection=0.2, refraction=0.7, refractive_index=1.5))
#         ],
#         'cubes': [
#             Cube(Vector3(-3, -1, -5), Vector3(-2, 0, -4), Material(Vector3(0.5, 0.5, 1), specular=0.4, reflection=0.1, metallic=0.6)),
#             Cube(Vector3(1, -1, -5), Vector3(2, 0, -4), Material(Vector3(1, 0.5, 0.5), specular=0.4, reflection=0.1, refraction=0.8, refractive_index=1.5))
#         ]
#     }
#
#
#     data = initialize_data(scene, camera)
#
#     # Add camera data to the data dictionary
#     data['camera'] = np.array([
#         camera['position'].x, camera['position'].y, camera['position'].z,
#         camera['look_at'].x, camera['look_at'].y, camera['look_at'].z,
#         camera['up'].x, camera['up'].y, camera['up'].z,
#         camera['fov']
#     ], dtype=np.float32)
#
#
#
#     # Camera setup (keep your existing camera setup here)
#     camera = {
#         'position': Vector3(-1, 1.5, 2),
#         'look_at': Vector3(0, 0, -5),
#         'up': Vector3(0, 1, 0)
#     }
#
#
#     if use_gpu:
#         # Get the directory of the current script
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#
#         # Construct the path to the CUDA kernel file
#         kernel_path = os.path.join(current_dir, 'ray_tracer_kernel.cu')
#
#         # Read the CUDA kernel code from the file
#         with open(kernel_path, 'r') as f:
#             cuda_code = f.read()
#
#         global ray_trace_kernel
#
#         # Create the RawKernel using the code from the file
#         ray_trace_kernel = cp.RawModule(code=cuda_code)
#         ray_trace_kernel = ray_trace_kernel.get_function('ray_trace_kernel')
#
#     print(f"Rendering at {width}x{height} with {samples} samples per pixel...")
#     start_time = time.time()
#
#     image = render(width, height, samples, data, camera, use_gpu)
#     end_time = time.time()
#
#     if image is None:
#         print("Rendering failed")
#         return
#
#     print(f"Final image shape: {image.shape}, dtype: {image.dtype}")
#     print(f"Image min: {np.min(image)}, max: {np.max(image)}")
#     print(f"Total time (including setup): {end_time - start_time:.2f} seconds")
#
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title("Ray Traced Scene")
#     plt.show()
#
#     # Save the image
#     plt.imsave("ray_traced_scene.png", image)
#     print("Image saved as ray_traced_scene.png")
def main():
    parser = argparse.ArgumentParser(description="Ray Tracing Renderer")
    parser.add_argument("--cpu", action="store_true", help="Force CPU rendering")
    parser.add_argument("--gpu", action="store_true", help="Force GPU rendering")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples per pixel")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")

    args = parser.parse_args()

    # Determine rendering mode
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = cp.cuda.is_available()

    width, height = args.width, args.height
    samples = args.samples

    print(f"Rendering at {width}x{height} with {samples} samples per pixel")
    print(f"Using {'GPU' if use_gpu else 'CPU'} rendering")

    # Camera setup
    # Represents position x, y, z, rotation u, v, w, and field of view fov
    camera_params = np.array([-1.0, 1.5, 2.0, 0.0, 0.0, -5.0, 60.0], dtype=np.float64)

    # Light setup
    light = np.array([5, 5, 5], dtype=np.float32)

 # Scene setup
    scene = {
        'spheres': np.array([
            [0, 1, -5, 1, 1, 0, 0, 0.6, 0.2, 0.8, 1.5, 0],  # Added metallic parameter
            [-2.5, 1, -7, 1.5, 0, 1, 0, 0.4, 0.8, 0, 1.0, 0.8],
            [2.5, 1, -6, 0.75, 0, 0, 1, 0.5, 0.1, 0.9, 1.5, 0]
        ], dtype=np.float32),
        'cylinders': np.array([
            [-1, 0, -4, 0.5, 2, 1, 0, 1, 0.7, 0.1, 0.8, 1.5, 0]  # Added metallic parameter
        ], dtype=np.float32),
        'planes': np.array([
            [0, 1, 0, 0, -1, 0, 0.5, 0.5, 0.5, 0.1, 0.1]
        ], dtype=np.float32),
        'rectangles': np.array([
            [1, 0, -4, 0.5, 0, 0, 0, 1, 0, 1, 0.5, 0, 0.3, 0.2, 0, 1, 0]  # Added refraction, refractive_index, metallic
        ], dtype=np.float32),
        'cubes': np.array([
            [-3, -1, -5, -2, 0, -4, 0.5, 0.5, 1, 0.4, 0.1, 0, 1.0, 0.6],
            [1, -1, -5, 2, 0, -4, 1, 0.5, 0.5, 0.4, 0.1, 0.8, 1.5, 0]
        ], dtype=np.float32)
    }

    # Prepare data for GPU
    gpu_data = {key: cp.array(value, dtype=cp.float32) for key, value in scene.items()}
    for key in scene.keys():
        gpu_data[f'num_{key}'] = np.int32(len(scene[key]))

    gpu_data['camera'] = cp.array(camera_params, dtype=cp.float64)
    gpu_data['light'] = cp.array(light, dtype=cp.float32)  # Example light position

    # Render
    start_time = time.time()
    if use_gpu:
        # Load CUDA kernel
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(current_dir, 'ray_tracer_kernel.cu')

        with open(kernel_path, 'r') as f:
            cuda_code = f.read()

        global ray_trace_kernel
        ray_trace_kernel = cp.RawModule(code=cuda_code).get_function('ray_trace_kernel')

    image = render(width, height, samples, scene, gpu_data, camera_params, use_gpu)

    end_time = time.time()

    print(f"Rendering completed in {end_time - start_time:.2f} seconds")

    # Display and save the image
    plt.imshow(image)
    plt.axis('off')
    plt.title("Ray Traced Scene")
    plt.show()

    plt.imsave("ray_traced_scene.png", image)
    print("Image saved as ray_traced_scene.png")

if __name__ == "__main__":
    main()
