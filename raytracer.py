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

def initialize_data(scene):
    data = {}

    data['spheres'] = np.array([
        [s.center.x, s.center.y, s.center.z, s.radius,
         s.material.color.x, s.material.color.y, s.material.color.z,
         s.material.specular, s.material.reflection, s.material.refraction, s.material.refractive_index]
        for s in scene['spheres']
    ], dtype=np.float32)

    data['cylinders'] = np.array([
        [c.center.x, c.center.y, c.center.z, c.radius, c.height,
         c.material.color.x, c.material.color.y, c.material.color.z,
         c.material.specular, c.material.reflection, c.material.refraction, c.material.refractive_index]
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
         r.material.specular, r.material.reflection]
        for r in scene['rectangles']
    ], dtype=np.float32)

    data['cubes'] = np.array([
        [c.min_point.x, c.min_point.y, c.min_point.z,
         c.max_point.x, c.max_point.y, c.max_point.z,
         c.material.color.x, c.material.color.y, c.material.color.z,
         c.material.specular, c.material.reflection, c.material.refraction, c.material.refractive_index]
        for c in scene['cubes']
    ], dtype=np.float32)

    if use_gpu:
        for key in data:
            data[key] = cp.array(data[key])

    return data

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflect(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal

def intersect_sphere(ray_origin, ray_direction, sphere):
    center = sphere[:3]
    radius = sphere[3]

    oc = ray_origin - center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - radius*radius
    discriminant = b*b - 4*a*c

    if discriminant < 0:
        return np.inf

    t = (-b - np.sqrt(discriminant)) / (2.0*a)
    return t if t > 0 else np.inf

def intersect_plane(ray_origin, ray_direction, plane):
    point = plane[:3]
    normal = plane[3:6]

    denom = np.dot(ray_direction, normal)
    if abs(denom) > 1e-6:
        t = np.dot(point - ray_origin, normal) / denom
        return t if t > 0 else np.inf
    return np.inf

def cpu_ray_color(ray_origin, ray_direction, scene_data, depth=0):
    if depth > 3:  # Limit recursion depth
        return np.zeros(3)

    closest_t = np.inf
    closest_obj = None

    # Check sphere intersections
    for sphere in scene_data['spheres']:
        t = intersect_sphere(ray_origin, ray_direction, sphere)
        if t < closest_t:
            closest_t = t
            closest_obj = (sphere, 'sphere')

    # Check plane intersections
    for plane in scene_data['planes']:
        t = intersect_plane(ray_origin, ray_direction, plane)
        if t < closest_t:
            closest_t = t
            closest_obj = (plane, 'plane')

    if closest_obj is None:
        # Sky color
        t = 0.5 * (ray_direction[1] + 1.0)
        return (1.0-t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])

    obj, obj_type = closest_obj
    hit_point = ray_origin + closest_t * ray_direction

    if obj_type == 'sphere':
        normal = normalize(hit_point - obj[:3])
        color = obj[4:7]
        specular = obj[7]
        reflection = obj[8]
    else:  # plane
        normal = obj[3:6]
        color = obj[6:9]
        specular = obj[9]
        reflection = obj[10]

    # Diffuse lighting
    light_dir = normalize(np.array([5, 5, 5]) - hit_point)
    diffuse = np.maximum(np.dot(normal, light_dir), 0)

    # Specular lighting
    reflect_dir = reflect(-light_dir, normal)
    spec = np.power(np.maximum(np.dot(-ray_direction, reflect_dir), 0), 50)
    specular_intensity = specular * spec

    # Reflection
    reflect_color = np.zeros(3)
    if reflection > 0 and depth < 3:
        reflect_dir = reflect(ray_direction, normal)
        reflect_origin = hit_point + normal * 0.001  # Offset to avoid self-intersection
        reflect_color = cpu_ray_color(reflect_origin, reflect_dir, scene_data, depth + 1)

    return color * (diffuse + 0.1) + specular_intensity + reflection * reflect_color

def render_chunk(chunk_data):
    y_start, y_end, width, height, samples, scene_data = chunk_data
    chunk = np.zeros((y_end - y_start, width, 3), dtype=np.float32)

    for j in range(y_start, y_end):
        for i in range(width):
            color = np.zeros(3)
            for _ in range(samples):
                u = (i + np.random.random()) / width
                v = (j + np.random.random()) / height
                ray_origin = np.array([0, 0, 0])
                ray_direction = normalize(np.array([(2*u - 1)*width/height, -(2*v - 1), -1]))

                color += cpu_ray_color(ray_origin, ray_direction, scene_data)

            color /= samples
            chunk[j - y_start, i] = np.clip(color, 0, 1)

    return chunk

def render_cpu(width, height, samples, data):
    num_cores = multiprocessing.cpu_count()
    chunk_size = height // num_cores

    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, height), width, height, samples, data)
        for i in range(num_cores)
    ]

    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(render_chunk, chunks), total=len(chunks), desc="Rendering"))

    return np.vstack(results)[::-1]  # Flip vertically

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

    # Check for CUDA errors
    cp.cuda.runtime.deviceSynchronize()
    error = cp.cuda.runtime.getLastError()
    if error != 0:
        print(f"CUDA error: {cp.cuda.runtime.getErrorString(error)}")
        return None

    return output

def render(width, height, samples, data):
    if use_gpu:
        return render_gpu(width, height, samples, data)
    else:
        return render_cpu(width, height, samples, data)

# Scene setup
scene = {
    'spheres': [
        Sphere(Vector3(0, 1, -5), 1, Material(Vector3(1, 0, 0), specular=0.6, reflection=0.2)),
        Sphere(Vector3(-2.5, 0.5, -7), 1.5, Material(Vector3(0, 1, 0), specular=0.4, reflection=0.8)),
        Sphere(Vector3(2.5, 0.5, -6), 0.75, Material(Vector3(0, 0, 1), specular=0.5, reflection=0.1))
    ],
    'cylinders': [
        Cylinder(Vector3(-1, 0, -4), 0.5, 1, Material(Vector3(1, 1, 0), specular=0.7, reflection=0.1)),
        Cylinder(Vector3(1, 0, -3), 0.5, 1, Material(Vector3(0, 1, 1), specular=0.7, reflection=0.1))
    ],
    'planes': [
        Plane(Vector3(0, -1, 0), Vector3(0, 1, 0), Material(Vector3(0.5, 0.5, 0.5), specular=0.1, reflection=0.1)),
        Plane(Vector3(0, 0, -10), Vector3(0, 0, 1), Material(Vector3(0.7, 0.7, 0.7), specular=0.1, reflection=0.1))
    ],
    'rectangles': [
        Rectangle(Vector3(-2, 2, -6), Vector3(2, 0, 0), Vector3(0, 2, 0), Material(Vector3(1, 0.5, 0), specular=0.3, reflection=0.2)),
        Rectangle(Vector3(2, -1, -4), Vector3(0, 2, 0), Vector3(2, 0, 0), Material(Vector3(0.5, 0, 1), specular=0.3, reflection=0.2))
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


    print(f"Rendering at {width}x{height} with {samples} samples per pixel...")
    start_time = time.time()

    image = render(width, height, samples, data)

    end_time = time.time()
    print(f"Total time (including setup): {end_time - start_time:.2f} seconds")

    if use_gpu:
        image = cp.asnumpy(image)

    plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Ray Traced Scene")
    plt.show()

    # Save the image
    plt.imsave("ray_traced_scene.png", image)
    print("Image saved as ray_traced_scene.png")

if __name__ == "__main__":
    main()

