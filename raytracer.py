import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import multiprocessing
from functools import partial

# Try to import CuPy, but fall back to NumPy if it's not available
try:
    import cupy as cp
    use_gpu = True
    print("GPU acceleration available. Using CuPy.")
except ImportError:
    use_gpu = False
    print("GPU acceleration not available. Using NumPy for CPU rendering.")

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Material:
    def __init__(self, color, specular=0, reflection=0, refraction=0, refractive_index=1):
        self.color = color
        self.specular = specular
        self.reflection = reflection
        self.refraction = refraction
        self.refractive_index = refractive_index

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
def object_to_array(obj):
    if isinstance(obj, (Sphere, Cylinder, Plane, Rectangle, Cube)):
        # Convert object attributes to a flat array
        attrs = []
        for attr in vars(obj).values():
            if isinstance(attr, Vector3):
                attrs.extend([attr.x, attr.y, attr.z])
            elif isinstance(attr, Material):
                attrs.extend([attr.color.x, attr.color.y, attr.color.z,
                              attr.specular, attr.reflection, attr.refraction,
                              attr.refractive_index])
            else:
                attrs.append(attr)
        return np.array(attrs, dtype=np.float32)
    elif isinstance(obj, Vector3):
        return np.array([obj.x, obj.y, obj.z], dtype=np.float32)
    else:
        return np.array(obj, dtype=np.float32)

def initialize_data(scene):
    data = {}
    for key, value in scene.items():
        if isinstance(value, list):
            data[key] = np.array([object_to_array(item) for item in value])
        else:
            data[key] = object_to_array(value)
    return data

# def initialize_data(scene):
#     data = {}
#
#     data['spheres'] = np.array([
#         [s.center.x, s.center.y, s.center.z, s.radius,
#          s.material.color.x, s.material.color.y, s.material.color.z,
#          s.material.specular, s.material.reflection, s.material.refraction, s.material.refractive_index]
#         for s in scene['spheres']
#     ], dtype=np.float32)
#
#     data['cylinders'] = np.array([
#         [c.center.x, c.center.y, c.center.z, c.radius, c.height,
#          c.material.color.x, c.material.color.y, c.material.color.z,
#          c.material.specular, c.material.reflection, c.material.refraction, c.material.refractive_index]
#         for c in scene['cylinders']
#     ], dtype=np.float32)
#
#     data['planes'] = np.array([
#         [p.point.x, p.point.y, p.point.z,
#          p.normal.x, p.normal.y, p.normal.z,
#          p.material.color.x, p.material.color.y, p.material.color.z,
#          p.material.specular, p.material.reflection]
#         for p in scene['planes']
#     ], dtype=np.float32)
#
#     data['rectangles'] = np.array([
#         [r.corner.x, r.corner.y, r.corner.z,
#          r.u.x, r.u.y, r.u.z,
#          r.v.x, r.v.y, r.v.z,
#          r.material.color.x, r.material.color.y, r.material.color.z,
#          r.material.specular, r.material.reflection]
#         for r in scene['rectangles']
#     ], dtype=np.float32)
#
#     data['cubes'] = np.array([
#         [c.min_point.x, c.min_point.y, c.min_point.z,
#          c.max_point.x, c.max_point.y, c.max_point.z,
#          c.material.color.x, c.material.color.y, c.material.color.z,
#          c.material.specular, c.material.reflection, c.material.refraction, c.material.refractive_index]
#         for c in scene['cubes']
#     ], dtype=np.float32)
#     #
#     if use_gpu:
#         for key in data:
#             data[key] = cp.array(data[key])
#     # for key, value in scene.items():
#     #     data[key] = np.array(value, dtype=np.float32)
#
#     return data

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflect(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal
def vector_add(a, b):
    return [a[i] + b[i] for i in range(3)]

def vector_subtract(a, b):
    return [a[i] - b[i] for i in range(3)]

def vector_multiply(v, scalar):
    return [v[i] * scalar for i in range(3)]

def vector_dot(a, b):
    return sum(a[i] * b[i] for i in range(3))

def vector_normalize(v):
    length = np.sqrt(sum(x**2 for x in v))
    return [x / length for x in v] if length != 0 else v

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
    return t if t > 0 else np.inf

def intersect_cylinder(ray_origin, ray_direction, cylinder):
    center = cylinder[:3]
    radius = cylinder[3]
    height = cylinder[4]
    oc = ray_origin - center
    a = ray_direction[0]**2 + ray_direction[2]**2
    b = 2 * (oc[0] * ray_direction[0] + oc[2] * ray_direction[2])
    c = oc[0]**2 + oc[2]**2 - radius**2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return np.inf
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    y = ray_origin[1] + t * ray_direction[1]
    if y < center[1] or y > center[1] + height:
        return np.inf
    return t if t > 0 else np.inf

def intersect_plane(ray_origin, ray_direction, plane):
    normal = plane[3:6]
    point = plane[:3]
    denom = np.dot(normal, ray_direction)
    if abs(denom) > 1e-6:
        t = np.dot(point - ray_origin, normal) / denom
        return t if t > 0 else np.inf
    return np.inf

def intersect_rectangle(ray_origin, ray_direction, rectangle):
    corner = rectangle[:3]
    u, v = rectangle[3:6], rectangle[6:9]
    normal = normalize(np.cross(u, v))
    t = np.dot(corner - ray_origin, normal) / np.dot(ray_direction, normal)
    if t < 0:
        return np.inf
    p = ray_origin + t * ray_direction
    pu = p - corner
    if 0 <= np.dot(pu, u) <= np.dot(u, u) and 0 <= np.dot(pu, v) <= np.dot(v, v):
        return t
    return np.inf

def intersect_cube(ray_origin, ray_direction, cube):
    min_point, max_point = cube[:3], cube[3:6]
    t_min = (min_point - ray_origin) / ray_direction
    t_max = (max_point - ray_origin) / ray_direction
    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_near = np.max(t1)
    t_far = np.min(t2)
    if t_near > t_far or t_far < 0:
        return np.inf
    return t_near if t_near > 0 else t_far

def is_in_shadow(hit_point, light_pos, scene_data):
    light_dir = normalize(light_pos - hit_point)
    for obj_type in ['spheres', 'cylinders', 'planes', 'rectangles', 'cubes']:
        for obj in scene_data[obj_type]:
            t = globals()[f"intersect_{obj_type[:-1]}"](hit_point, light_dir, obj)
            if 0 < t < np.linalg.norm(light_pos - hit_point):
                return True
    return False

def trace_ray(ray_origin, ray_direction, spheres, planes, depth=0):
    if depth > 5:
        return [0, 0, 0]

    closest_t = float('inf')
    closest_obj = None
    closest_type = None

    for sphere in spheres:
        t = intersect_sphere(ray_origin, ray_direction, sphere)
        if t and t < closest_t:
            closest_t = t
            closest_obj = sphere
            closest_type = 'sphere'

    for plane in planes:
        t = intersect_plane(ray_origin, ray_direction, plane)
        if t and t < closest_t:
            closest_t = t
            closest_obj = plane
            closest_type = 'plane'

    if closest_obj is None:
        return [0.5, 0.7, 1.0]  # Sky color

    hit_point = vector_add(ray_origin, vector_multiply(ray_direction, closest_t))

    if closest_type == 'sphere':
        normal = vector_normalize(vector_subtract(hit_point, closest_obj[:3]))
        color = closest_obj[4:7]
    else:  # plane
        normal = closest_obj[3:6]
        color = closest_obj[6:9]

    light_dir = vector_normalize([1, 1, -1])  # Directional light
    light_intensity = max(0, vector_dot(normal, light_dir))

    reflection = closest_obj[-2]
    if reflection > 0 and depth < 5:
        reflect_dir = vector_subtract(ray_direction, vector_multiply(normal, 2 * vector_dot(ray_direction, normal)))
        reflect_color = trace_ray(hit_point, reflect_dir, spheres, planes, depth + 1)
        color = vector_add(vector_multiply(color, 1 - reflection), vector_multiply(reflect_color, reflection))

    return vector_multiply(color, light_intensity)

def render_pixel(x, y, width, height, spheres, planes, samples=1):
    color = [0, 0, 0]
    for _ in range(samples):
        u = (x + np.random.random()) / width
        v = (y + np.random.random()) / height
        ray_direction = vector_normalize([(u - 0.5) * width / height, (0.5 - v), -1])
        ray_origin = [0, 0, 0]
        pixel_color = trace_ray(ray_origin, ray_direction, spheres, planes)
        color = vector_add(color, pixel_color)
    return vector_multiply(color, 1 / samples)

def render_chunk(args):
    x_start, x_end, y_start, y_end, width, height, scene_data, samples = args
    chunk = np.zeros((y_end - y_start, x_end - x_start, 3), dtype=np.float32)
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            color = np.zeros(3)
            for _ in range(samples):
                u = (x + np.random.random()) / width
                v = (y + np.random.random()) / height
                ray_direction = normalize(np.array([(u - 0.5) * width / height, (0.5 - v), -1]))
                ray_origin = np.array([0, 0, 0])
                color += cpu_ray_color(ray_origin, ray_direction, scene_data)
            chunk[y - y_start, x - x_start] = color / samples
    return chunk

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
        refraction = closest_obj[9]
        refractive_index = closest_obj[10]
    elif obj_type == 'cylinder':
        center = closest_obj[:3]
        normal = normalize(np.array([hit_point[0] - center[0], 0, hit_point[2] - center[2]]))
        color = closest_obj[5:8]
        specular = closest_obj[8]
        reflection = closest_obj[9]
        refraction = closest_obj[10]
        refractive_index = closest_obj[11]
    elif obj_type == 'plane':
        normal = closest_obj[3:6]
        if np.dot(normal, ray_direction) > 0:
            normal = -normal
        x, z = hit_point[0], hit_point[2]
        color = closest_obj[6:9] if (int(x * 2) % 2 == int(z * 2) % 2) else closest_obj[6:9] * 0.8
        specular = closest_obj[9]
        reflection = closest_obj[10]
        refraction = 0
        refractive_index = 1
    elif obj_type == 'rectangle':
        u, v = closest_obj[3:6], closest_obj[6:9]
        normal = normalize(np.cross(u, v))
        if np.dot(normal, ray_direction) > 0:
            normal = -normal
        color = closest_obj[9:12]
        specular = closest_obj[12]
        reflection = closest_obj[13]
        refraction = 0
        refractive_index = 1
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
        specular = closest_obj[9]
        reflection = closest_obj[10]
        refraction = closest_obj[11]
        refractive_index = closest_obj[12]

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

    color = ambient + (diffuse + specular_intensity) * shadow_factor

    # Reflection
    if reflection > 0 and depth < 5:
        reflect_dir = reflect(ray_direction, normal)
        reflect_origin = hit_point + normal * 0.001
        reflect_color = cpu_ray_color(reflect_origin, reflect_dir, scene_data, depth + 1)
        color = color * (1 - reflection) + reflect_color * reflection

    # Refraction (simplified, only for spheres and cylinders)
    if refraction > 0 and depth < 5:
        refract_dir = refract(ray_direction, normal, 1.0, refractive_index)
        if refract_dir is not None:
            refract_origin = hit_point - normal * 0.001
            refract_color = cpu_ray_color(refract_origin, refract_dir, scene_data, depth + 1)
            color = color * (1 - refraction) + refract_color * refraction

    return np.clip(color, 0, 1)

def refract(incident, normal, n1, n2):
    cos_i = -np.dot(normal, incident)
    eta = n1 / n2
    k = 1 - eta * eta * (1 - cos_i * cos_i)
    if k < 0:
        return None
    return eta * incident + (eta * cos_i - np.sqrt(k)) * normal

#
# def render_chunk(chunk_data):
#     y_start, y_end, width, height, samples, scene_data = chunk_data
#     chunk = np.zeros((y_end - y_start, width, 3), dtype=np.float32)
#
#     for j in range(y_start, y_end):
#         for i in range(width):
#             color = np.zeros(3)
#             for _ in range(samples):
#                 u = (i + np.random.random()) / width
#                 v = (j + np.random.random()) / height
#                 ray_origin = np.array([0, 0, 0])
#                 ray_direction = normalize(np.array([(2*u - 1)*width/height, -(2*v - 1), -1]))
#
#                 color += cpu_ray_color(ray_origin, ray_direction, scene_data)
#
#             color /= samples
#             chunk[j - y_start, i] = np.clip(color, 0, 1)


# def render_cpu(width, height, samples, data):
#     # Ensure all data is in NumPy format
#     data = {k: np.array(v) for k, v in data.items()}
#
#     num_threads = multiprocessing.cpu_count()
#     chunk_height = height // num_threads
#
#     pool = multiprocessing.Pool(processes=num_threads)
#     chunks = []
#     for i in range(num_threads):
#         y_start = i * chunk_height
#         y_end = y_start + chunk_height if i < num_threads - 1 else height
#         chunks.append((0, width, y_start, y_end, width, height, data, samples))
#
#     results = list(tqdm(pool.imap(render_chunk, chunks), total=len(chunks), desc="Rendering"))
#
#     return np.vstack(results)
def render_cpu(width, height, samples, data):
    num_threads = multiprocessing.cpu_count()
    chunk_height = height // num_threads

    pool = multiprocessing.Pool(processes=num_threads)
    chunks = []
    for i in range(num_threads):
        y_start = i * chunk_height
        y_end = y_start + chunk_height if i < num_threads - 1 else height
        chunks.append((0, width, y_start, y_end, width, height, data, samples))

    results = list(tqdm(pool.imap(render_chunk, chunks), total=len(chunks), desc="Rendering"))

    return np.vstack(results)

def render_gpu(width, height, samples, data):
    output = cp.zeros((height, width, 3), dtype=cp.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    ray_trace_kernel(
        grid=blockspergrid,
        block=threadsperblock,
        args=(output, width, height, samples,
              data['spheres'], data['spheres'].shape[0],
              data['cylinders'], data['cylinders'].shape[0],
              data['planes'], data['planes'].shape[0],
              data['rectangles'], data['rectangles'].shape[0],
              data['cubes'], data['cubes'].shape[0])
    )

    cp.cuda.stream.get_current_stream().synchronize()
    return output

def render(width, height, samples, data):
    if use_gpu:
        try:
            # Convert data to CuPy arrays for GPU rendering
            gpu_data = {k: cp.array(v) for k, v in data.items()}
            image = render_gpu(width, height, samples, gpu_data)
            if image is None:
                print("GPU rendering failed, falling back to CPU")
                image = render_cpu(width, height, samples, data)
            else:
                # Convert CuPy array to NumPy array
                image = cp.asnumpy(image)
        except Exception as e:
            print(f"Error during GPU rendering: {e}")
            print("Falling back to CPU rendering")
            image = render_cpu(width, height, samples, data)
    else:
        image = render_cpu(width, height, samples, data)

    # Ensure the image is in the correct format
    image = np.clip(image, 0, 1)  # Clip values between 0 and 1
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit format
    return image

# Scene setup
scene = {
    'spheres': [
        Sphere(Vector3(0, 1, -5), 1, Material(Vector3(1, 0, 0), specular=0.6, reflection=0.2)),
        Sphere(Vector3(-2.5, 1, -7), 1.5, Material(Vector3(0, 1, 0), specular=0.4, reflection=0.8)),
        Sphere(Vector3(2.5, 1, -6), 0.75, Material(Vector3(0, 0, 1), specular=0.5, reflection=0.1))
    ],
    'cylinders': [
        Cylinder(Vector3(-1, 0, -4), 0.3, 1, Material(Vector3(1, 0, 1), specular=0.7, reflection=0.1, refraction=0.9, refractive_index=1.5))  # Glass cylinder tinted magenta
    ],
    'planes': [
        Plane(Vector3(0, 0, -10), Vector3(0, 0, 1), Material(Vector3(0.5, 0.5, 0.5), specular=0.1, reflection=0.1))  # Back wall
    ],
    'rectangles': [
        Rectangle(Vector3(1, 0, -4), Vector3(0.5, 0, 0), Vector3(0, 1, 0), Material(Vector3(1, 0.5, 0), specular=0.3, reflection=0.2))
    ],
    'cubes': [
        Cube(Vector3(-1, -1, -3), Vector3(0, 0, -2), Material(Vector3(0.5, 0.5, 1), specular=0.4, reflection=0.1)),
        Cube(Vector3(1, 1, -5), Vector3(2, 2, -4), Material(Vector3(1, 0.5, 0.5), specular=0.4, reflection=0.1))
    ]
}

def main():
    width, height = 800, 600
    samples = 4

    data = initialize_data(scene)

    if use_gpu:
        import os

        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the CUDA kernel file
        kernel_path = os.path.join(current_dir, 'ray_tracer_kernel.cu')

        # Read the CUDA kernel code from the file
        with open(kernel_path, 'r') as f:
            cuda_code = f.read()

        global ray_trace_kernel

        # Create the RawKernel using the code from the file
        ray_trace_kernel = cp.RawKernel(cuda_code, 'ray_trace_kernel')


    # print("Data shapes:")
    # for key, value in data.items():
    #     print(f"{key}: {value.shape}")

    print(f"Rendering at {width}x{height} with {samples} samples per pixel...")
    start_time = time.time()

    image = render(width, height, samples, data)
    end_time = time.time()

    if image is None:
        print("Rendering failed")
        return

    # At this point, 'image' should be a NumPy array in 8-bit format
    print(f"Final image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image min: {np.min(image)}, max: {np.max(image)}")
    print(f"Total time (including setup): {end_time - start_time:.2f} seconds")

    # if use_gpu:
    #     image = cp.asnumpy(image)

    # plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Ray Traced Scene")
    plt.show()

    # Save the image
    plt.imsave("ray_traced_scene.png", image)
    print("Image saved as ray_traced_scene.png")

if __name__ == "__main__":
    main()

