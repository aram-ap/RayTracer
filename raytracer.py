import numpy as np
import matplotlib.pyplot as plt
import random
import math
import multiprocessing
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import time
from tqdm import tqdm
import logging
# At the beginning of your script, replace the CuPy import with:
try:
    import cupy as cp
    xp = cp
    using_gpu = True
    logging.info("Using GPU acceleration")
except ImportError:
    xp = np
    using_gpu = False
    logging.info("GPU acceleration not available, using CPU")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use GPU if available, otherwise use CPU
try:
    xp = cp.get_array_module(cp.array([1]))
    logging.info("Using GPU acceleration")
except ImportError:
    xp = np
    logging.info("GPU acceleration not available, using CPU")

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def normalize(self):
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector3(self.x / length, self.y / length, self.z / length)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

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
        self.normal = normal.normalize()
        self.material = material

class Light:
    def __init__(self, position, color, intensity):
        self.position = position
        self.color = color
        self.intensity = intensity

class AreaLight:
    def __init__(self, position, u, v, color, intensity, samples=4):
        self.position = position
        self.u = u
        self.v = v
        self.color = color
        self.intensity = intensity
        self.samples = samples

def checkered_pattern(point, scale=1.0):
    x = math.floor(point.x * scale)
    z = math.floor(point.z * scale)
    return (x + z) % 2 == 0

# Scene setup
scene = {
    'global_light': Light(Vector3(0, 10, 10), Vector3(1, 1, 1), 0.2),
    'area_light': AreaLight(Vector3(5, 5, 5), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(1, 1, 1), 0.8, samples=16),
    'spheres': [
        Sphere(Vector3(0, 1, -5), 1, Material(Vector3(1, 0, 0), specular=0.6, reflection=0.2)),
        Sphere(Vector3(-2.5, 1, -7), 1.5, Material(Vector3(0, 1, 0), specular=0.4, reflection=1.0)),  # Mirror sphere
        Sphere(Vector3(2.5, 1, -6), 0.75, Material(Vector3(0, 0, 1), specular=0.5, reflection=0.1))
    ],
    'cylinders': [
        Cylinder(Vector3(-1, 0.5, -4), 0.5, 1, Material(Vector3(1, 1, 1), specular=0.7, reflection=0.1, refraction=0.9, refractive_index=1.5)),
        Cylinder(Vector3(1, 0.5, -3), 0.5, 1, Material(Vector3(1, 1, 1), specular=0.7, reflection=0.1, refraction=0.9, refractive_index=1.5))
    ],
    'planes': [
        Plane(Vector3(0, 0, 0), Vector3(0, 1, 0), Material(Vector3(0.8, 0.8, 0.8))),  # Floor
        Plane(Vector3(0, 0, -10), Vector3(0, 0, 1), Material(Vector3(0.8, 0.8, 0.8))),  # Back wall
        Plane(Vector3(-10, 0, 0), Vector3(1, 0, 0), Material(Vector3(0.8, 0.8, 0.8))),  # Left wall
        Plane(Vector3(10, 0, 0), Vector3(-1, 0, 0), Material(Vector3(0.8, 0.8, 0.8)))   # Right wall
    ]
}

def intersect_sphere(ray, sphere):
    oc = ray.origin - sphere.center
    a = ray.direction.dot(ray.direction)
    b = 2 * oc.dot(ray.direction)
    c = oc.dot(oc) - sphere.radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    t = (-b - math.sqrt(discriminant)) / (2*a)
    if t < 0:
        return None
    return t

def intersect_cylinder(ray, cylinder):
    oc = ray.origin - cylinder.center
    a = ray.direction.x**2 + ray.direction.z**2
    b = 2 * (oc.x * ray.direction.x + oc.z * ray.direction.z)
    c = oc.x**2 + oc.z**2 - cylinder.radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    t = (-b - math.sqrt(discriminant)) / (2*a)
    if t < 0:
        return None
    y = ray.origin.y + t * ray.direction.y
    if y < cylinder.center.y or y > cylinder.center.y + cylinder.height:
        return None
    return t

def intersect_plane(ray, plane):
    denom = ray.direction.dot(plane.normal)
    if abs(denom) > 1e-6:
        t = (plane.point - ray.origin).dot(plane.normal) / denom
        if t > 0:
            return t
    return None

def find_nearest_intersection(ray, scene):
    nearest_intersection = None
    min_distance = float('inf')

    for obj in scene['spheres'] + scene['cylinders'] + scene['planes']:
        if isinstance(obj, Sphere):
            t = intersect_sphere(ray, obj)
        elif isinstance(obj, Cylinder):
            t = intersect_cylinder(ray, obj)
        else:  # Plane
            t = intersect_plane(ray, obj)

        if t and t < min_distance:
            min_distance = t
            if isinstance(obj, Sphere):
                point = ray.origin + ray.direction * t
                normal = (point - obj.center).normalize()
            elif isinstance(obj, Cylinder):
                point = ray.origin + ray.direction * t
                normal = Vector3(point.x - obj.center.x, 0, point.z - obj.center.z).normalize()
            else:  # Plane
                point = ray.origin + ray.direction * t
                normal = obj.normal
            nearest_intersection = (obj, point, normal)

    return nearest_intersection

def compute_lighting(point, normal, view_dir, material, scene):
    color = Vector3(0, 0, 0)
    for light in [scene['global_light'], scene['area_light']]:
        if isinstance(light, AreaLight):
            light_contribution = Vector3(0, 0, 0)
            for _ in range(light.samples):
                random_u = random.random() - 0.5
                random_v = random.random() - 0.5
                light_pos = light.position + light.u * random_u + light.v * random_v
                light_dir = (light_pos - point).normalize()
                shadow_ray = Ray(point + normal * 0.001, light_dir)
                shadow_intersection = find_nearest_intersection(shadow_ray, scene)

                if not shadow_intersection or (shadow_intersection[0].material.refraction > 0):
                    diffuse = max(0, normal.dot(light_dir))
                    light_contribution = light_contribution + light.color * light.intensity * diffuse

                    if material.specular > 0:
                        reflect_dir = light_dir - normal * (2 * light_dir.dot(normal))
                        specular = max(0, view_dir.dot(reflect_dir)) ** 50
                        light_contribution = light_contribution + light.color * light.intensity * material.specular * specular

            color = color + light_contribution * (1 / light.samples)
        else:
            light_dir = (light.position - point).normalize()
            shadow_ray = Ray(point + normal * 0.001, light_dir)
            shadow_intersection = find_nearest_intersection(shadow_ray, scene)

            if not shadow_intersection or (shadow_intersection[0].material.refraction > 0):
                diffuse = max(0, normal.dot(light_dir))
                color = color + material.color * light.color * light.intensity * diffuse

                if material.specular > 0:
                    reflect_dir = light_dir - normal * (2 * light_dir.dot(normal))
                    specular = max(0, view_dir.dot(reflect_dir)) ** 50
                    color = color + light.color * light.intensity * material.specular * specular

    return color

def compute_refraction(ray, hit_point, normal, material, scene, depth):
    n1 = 1.0  # Air refractive index
    n2 = material.refractive_index

    cos_i = -normal.dot(ray.direction)
    if cos_i < 0:
        cos_i = -cos_i
        normal = normal * -1
        n1, n2 = n2, n1

    n = n1 / n2
    k = 1 - n * n * (1 - cos_i * cos_i)
    if k < 0:
        return Vector3(0, 0, 0)  # Total internal reflection

    refract_dir = (ray.direction * n + normal * (n * cos_i - math.sqrt(k))).normalize()
    refract_ray = Ray(hit_point - normal * 0.001, refract_dir)
    return trace_ray(refract_ray, scene, depth + 1)

def trace_ray(ray, scene, depth=0):
    if depth > 5:
        return Vector3(0, 0, 0)

    intersection = find_nearest_intersection(ray, scene)
    if not intersection:
        return Vector3(0, 0, 0)

    obj, hit_point, normal = intersection
    material = obj.material

    if isinstance(obj, Plane):
        if checkered_pattern(hit_point):
            color = Vector3(0.2, 0.2, 0.2)  # Dark gray
        else:
            color = Vector3(0.8, 0.8, 0.8)  # Light gray
    else:
        color = material.color

    # Compute reflection
    if material.reflection > 0:
        reflect_dir = ray.direction - normal * (2 * ray.direction.dot(normal))
        reflect_ray = Ray(hit_point + normal * 0.001, reflect_dir)
        reflect_color = trace_ray(reflect_ray, scene, depth + 1)
        color = color * (1 - material.reflection) + reflect_color * material.reflection

    # Compute refraction
    if material.refraction > 0:
        refract_color = compute_refraction(ray, hit_point, normal, material, scene, depth)
        color = color * (1 - material.refraction) + refract_color * material.refraction

    # Compute lighting
    light_color = compute_lighting(hit_point, normal, ray.direction * -1, material, scene)
    color = color * light_color

    return color
def render_pixel(x, y, width, height, scene, samples=1):
    aspect_ratio = width / height
    color = Vector3(0, 0, 0)
    for _ in range(samples):
        u = ((x + random.random()) / width) * 2 - 1
        v = -((y + random.random()) / height * 2 - 1) / aspect_ratio
        ray = Ray(Vector3(0, 0, 0), Vector3(u, v, -1).normalize())
        color = color + trace_ray(ray, scene)
    return color * (1 / samples)

def render_chunk(args):
    x_start, x_end, y_start, y_end, width, height, scene, samples = args
    chunk = np.zeros((y_end - y_start, x_end - x_start, 3))
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            color = render_pixel(x, y, width, height, scene, samples)
            chunk[y - y_start, x - x_start] = [min(1, max(0, c)) for c in (color.x, color.y, color.z)]
    return chunk

def render(width, height, samples=4):
    num_threads = multiprocessing.cpu_count()
    chunk_height = height // num_threads

    pool = multiprocessing.Pool(processes=num_threads)
    chunks = []
    for i in range(num_threads):
        y_start = i * chunk_height
        y_end = y_start + chunk_height if i < num_threads - 1 else height
        chunks.append((0, width, y_start, y_end, width, height, scene, samples))

    results = list(tqdm(pool.imap(render_chunk, chunks), total=len(chunks), desc="Rendering"))

    image = np.vstack(results)

    if using_gpu:
        image = cp.array(image)

    return (image * 255).astype(xp.uint8)

window = None

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(-1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, 1)
    glTexCoord2f(0, 0); glVertex2f(-1, 1)
    glEnd()
    glDisable(GL_TEXTURE_2D)

    glutSwapBuffers()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    if key == b'\x1b':  # ESC key
        glutDestroyWindow(window)
        sys.exit()

# In the main function, modify the image export:
def export_image(image, filename="render.png"):
    if using_gpu:
        image = cp.asnumpy(image)
    Image.fromarray(image).save(filename)
    logging.info(f"Image exported as {filename}")

def main():
    width, height = 800, 600
    samples = 4

    logging.info(f"Rendering at {width}x{height} with {samples} samples per pixel...")
    start_time = time.time()

    image = render(width, height, samples)

    end_time = time.time()
    logging.info(f"Rendering complete. Time taken: {end_time - start_time:.2f} seconds")

    if using_gpu:
        image_cpu = cp.asnumpy(image)
    else:
        image_cpu = image

    # Save the image
    plt.imsave("render.png", image_cpu.astype(np.uint8))
    logging.info("Image saved as render.png")

    # Display the image
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image_cpu.astype(np.uint8))
    plt.axis('off')
    plt.title("Ray Traced Scene")
    plt.show()

if __name__ == "__main__":
    main()

