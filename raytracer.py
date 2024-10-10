import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

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

def cpu_ray_color(ray, scene_data):
    # This is a very basic implementation. You should expand this for better results.
    for sphere in scene_data['spheres']:
        center = sphere[:3]
        radius = sphere[3]
        color = sphere[4:7]

        oc = ray[0] - center
        a = np.dot(ray[1], ray[1])
        b = 2.0 * np.dot(oc, ray[1])
        c = np.dot(oc, oc) - radius*radius
        discriminant = b*b - 4*a*c

        if discriminant > 0:
            return color

    # If no intersection, return background color
    t = 0.5 * (ray[1][1] + 1.0)
    return (1.0-t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])


def render_cpu(width, height, samples, data):
    output = np.zeros((height, width, 3), dtype=np.float32)

    for j in tqdm(range(height), desc="Rendering", unit="lines"):
        for i in range(width):
            color = np.zeros(3)
            for _ in range(samples):
                u = (i + np.random.random()) / width
                v = (j + np.random.random()) / height
                ray_origin = np.array([0, 0, 0])
                ray_direction = np.array([(2*u - 1)*width/height, -(2*v - 1), -1])
                ray_direction /= np.linalg.norm(ray_direction)

                color += cpu_ray_color([ray_origin, ray_direction], data)

            color /= samples
            output[height-j-1, i] = np.clip(color, 0, 1)

    return output

def render_gpu(width, height, samples, data):
    output = cp.zeros((height, width, 3), dtype=cp.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Create a CUDA event to measure render time
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()

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

    end_event.record()
    end_event.synchronize()

    if cp.cuda.runtime.getLastError() != 0:
        print("CUDA error: {}".format(cp.cuda.runtime.getLastError()))

    render_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000  # Convert to seconds
    print(f"GPU rendering completed in {render_time:.2f} seconds")

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
        # Load and compile the CUDA kernel
        kernel_path = os.path.join(os.path.dirname(__file__), 'ray_tracer_kernel.cu')
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        global ray_trace_kernel
        ray_trace_kernel = cp.RawKernel(kernel_code, 'ray_trace_kernel')

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

