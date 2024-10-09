import numpy as np
import random
import math
import multiprocessing
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image

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

    def normalize(self):
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector3(self.x / length, self.y / length, self.z / length)

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

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

class Cube:
    def __init__(self, min_point, max_point, material):
        self.min_point = min_point
        self.max_point = max_point
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

# Scene setup
scene = {
    'global_light': Light(Vector3(0, 10, 10), Vector3(1, 1, 1), 0.2),
    'area_light': AreaLight(Vector3(5, 5, 5), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(1, 1, 1), 0.8, samples=16),
    'spheres': [
        Sphere(Vector3(0, 0, -5), 1, Material(Vector3(1, 0, 0), specular=0.6, reflection=0.2)),
        Sphere(Vector3(-2.5, 0, -7), 1.5, Material(Vector3(0, 1, 0), specular=0.4, reflection=0.3)),
        Sphere(Vector3(2.5, 0, -6), 0.75, Material(Vector3(0, 0, 1), specular=0.5, reflection=0.1))
    ],
    'cube': Cube(Vector3(-0.5, -0.5, -4), Vector3(0.5, 0.5, -3),
                 Material(Vector3(1, 1, 1), specular=0.7, reflection=0.1, refraction=0.9, refractive_index=1.5)),
    'box': Cube(Vector3(-10, -10, -20), Vector3(10, 10, 0),
                Material(Vector3(0.8, 0.8, 0.8), specular=0.1, reflection=0.1))
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

def intersect_cube(ray, cube):
    t_min = float('-inf')
    t_max = float('inf')

    for i in range(3):
        if i == 0:
            min_val, max_val = cube.min_point.x, cube.max_point.x
            origin = ray.origin.x
            direction = ray.direction.x
        elif i == 1:
            min_val, max_val = cube.min_point.y, cube.max_point.y
            origin = ray.origin.y
            direction = ray.direction.y
        else:
            min_val, max_val = cube.min_point.z, cube.max_point.z
            origin = ray.origin.z
            direction = ray.direction.z

        if abs(direction) < 1e-8:  # Check for near-zero direction
            if origin < min_val or origin > max_val:
                return None
        else:
            t1 = (min_val - origin) / direction
            t2 = (max_val - origin) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

    return t_min if t_min > 0 else t_max

def sphere_normal(sphere, point):
    return (point - sphere.center).normalize()

def cube_normal(cube, point):
    epsilon = 1e-5
    if abs(point.x - cube.min_point.x) < epsilon: return Vector3(-1, 0, 0)
    if abs(point.x - cube.max_point.x) < epsilon: return Vector3(1, 0, 0)
    if abs(point.y - cube.min_point.y) < epsilon: return Vector3(0, -1, 0)
    if abs(point.y - cube.max_point.y) < epsilon: return Vector3(0, 1, 0)
    if abs(point.z - cube.min_point.z) < epsilon: return Vector3(0, 0, -1)
    if abs(point.z - cube.max_point.z) < epsilon: return Vector3(0, 0, 1)
    return Vector3(0, 1, 0)  # Default case, shouldn't happen

def find_nearest_intersection(ray, scene):
    nearest_intersection = None
    min_distance = float('inf')

    for obj in scene['spheres'] + [scene['cube'], scene['box']]:
        if isinstance(obj, Sphere):
            t = intersect_sphere(ray, obj)
        else:
            t = intersect_cube(ray, obj)

        if t and t < min_distance:
            min_distance = t
            if isinstance(obj, Sphere):
                point = ray.origin + ray.direction * t
                normal = sphere_normal(obj, point)
            else:
                point = ray.origin + ray.direction * t
                normal = cube_normal(obj, point)
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
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        if y % 10 == 0:
            logging.debug(f"Rendered line {y} in chunk ({x_start}, {y_start}) to ({x_end}, {y_end})")
    return chunk

def render(width, height, samples=1):
    logging.info(f"Starting render: {width}x{height} with {samples} samples per pixel")
    start_time = time.time()

    image = np.zeros((height, width, 3))
    for y in tqdm(range(height), desc="Rendering", unit="line"):
        for x in range(width):
            color = render_pixel(x, y, width, height, scene, samples)
            image[y, x] = [min(1, max(0, c)) for c in (color.x, color.y, color.z)]
        if y % 10 == 0:
            logging.debug(f"Rendered line {y}")

    end_time = time.time()
    logging.info(f"Render completed in {end_time - start_time:.2f} seconds")

    return (image * 255).astype(np.uint8)

def main():
    # Start with a very low resolution for testing
    width, height = 200, 150
    samples = 1  # Use only 1 sample per pixel for now

    logging.info(f"Rendering at {width}x{height} with {samples} samples per pixel...")

    image = render(width, height, samples)

    logging.info("Rendering complete. Displaying image...")

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"Ray Traced Scene")

    glEnable(GL_TEXTURE_2D)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMainLoop()

if __name__ == "__main__":
    main()

