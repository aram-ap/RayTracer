#include <curand_kernel.h>

// Structures
struct Vector3 {
    float x, y, z;
};

struct Ray {
    Vector3 origin;
    Vector3 direction;
};

struct Material {
    Vector3 color;
    float specular;
    float reflection;
    float refraction;
    float refractive_index;
};

struct Sphere {
    Vector3 center;
    float radius;
    Material material;
};

struct Cylinder {
    Vector3 center;
    float radius;
    float height;
    Material material;
};

struct Plane {
    Vector3 point;
    Vector3 normal;
    Material material;
};

struct Rectangle {
    Vector3 corner;
    Vector3 u;
    Vector3 v;
    Material material;
};

struct Cube {
    Vector3 min_point;
    Vector3 max_point;
    Material material;
};

// Vector operations
__device__ Vector3 vector_add(Vector3 a, Vector3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ Vector3 vector_subtract(Vector3 a, Vector3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Vector3 vector_multiply(Vector3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__device__ Vector3 vector_multiply_vec(Vector3 a, Vector3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float vector_dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vector3 vector_cross(Vector3 a, Vector3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ Vector3 vector_normalize(Vector3 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0) {
        return {v.x / length, v.y / length, v.z / length};
    }
    return v;
}

// Intersection functions
__device__ bool intersect_sphere(Ray ray, Sphere sphere, float* t) {
    Vector3 oc = vector_subtract(ray.origin, sphere.center);
    float a = vector_dot(ray.direction, ray.direction);
    float b = 2.0f * vector_dot(oc, ray.direction);
    float c = vector_dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
        float temp = (-b - sqrtf(discriminant)) / (2.0f * a);
        if (temp > 0.001f) {
            *t = temp;
            return true;
        }
        temp = (-b + sqrtf(discriminant)) / (2.0f * a);
        if (temp > 0.001f) {
            *t = temp;
            return true;
        }
    }
    return false;
}

__device__ bool intersect_cylinder(Ray ray, Cylinder cylinder, float* t) {
    Vector3 ro = vector_subtract(ray.origin, cylinder.center);
    float a = ray.direction.x * ray.direction.x + ray.direction.z * ray.direction.z;
    float b = 2 * (ro.x * ray.direction.x + ro.z * ray.direction.z);
    float c = ro.x * ro.x + ro.z * ro.z - cylinder.radius * cylinder.radius;

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;

    float t0 = (-b - sqrtf(discriminant)) / (2 * a);
    float t1 = (-b + sqrtf(discriminant)) / (2 * a);

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    float y0 = ro.y + t0 * ray.direction.y;
    float y1 = ro.y + t1 * ray.direction.y;

    float cylinder_y_min = -cylinder.height / 2;
    float cylinder_y_max = cylinder.height / 2;

    if (y0 < cylinder_y_min) {
        if (y1 < cylinder_y_min) return false;
        float th = t0 + (t1 - t0) * (cylinder_y_min - y0) / (y1 - y0);
        if (th > 0) {
            *t = th;
            return true;
        }
    } else if (y0 >= cylinder_y_min && y0 <= cylinder_y_max) {
        if (t0 > 0) {
            *t = t0;
            return true;
        }
    }

    return false;
}

__device__ bool intersect_plane(Ray ray, Plane plane, float* t) {
    float denom = vector_dot(plane.normal, ray.direction);
    if (fabsf(denom) > 1e-6) {
        Vector3 p0l0 = vector_subtract(plane.point, ray.origin);
        *t = vector_dot(p0l0, plane.normal) / denom;
        return (*t >= 0);
    }
    return false;
}

__device__ bool intersect_rectangle(Ray ray, Rectangle rect, float* t) {
    Vector3 n = vector_normalize(vector_cross(rect.u, rect.v));
    float denom = vector_dot(n, ray.direction);

    if (fabsf(denom) < 1e-6) return false;

    Vector3 p0r0 = vector_subtract(rect.corner, ray.origin);
    *t = vector_dot(p0r0, n) / denom;

    if (*t < 0) return false;

    Vector3 p = vector_add(ray.origin, vector_multiply(ray.direction, *t));
    Vector3 vi = vector_subtract(p, rect.corner);

    float a1 = vector_dot(vi, rect.u);
    if (a1 < 0 || a1 > vector_dot(rect.u, rect.u)) return false;

    float a2 = vector_dot(vi, rect.v);
    if (a2 < 0 || a2 > vector_dot(rect.v, rect.v)) return false;

    return true;
}

__device__ bool intersect_cube(Ray ray, Cube cube, float* t) {
    Vector3 inv_dir = {1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};
    Vector3 t_min = vector_multiply_vec(vector_subtract(cube.min_point, ray.origin), inv_dir);
    Vector3 t_max = vector_multiply_vec(vector_subtract(cube.max_point, ray.origin), inv_dir);

    Vector3 t_near = {fminf(t_min.x, t_max.x), fminf(t_min.y, t_max.y), fminf(t_min.z, t_max.z)};
    Vector3 t_far = {fmaxf(t_min.x, t_max.x), fmaxf(t_min.y, t_max.y), fmaxf(t_min.z, t_max.z)};

    float t_near_max = fmaxf(fmaxf(t_near.x, t_near.y), t_near.z);
    float t_far_min = fminf(fminf(t_far.x, t_far.y), t_far.z);

    if (t_near_max > t_far_min || t_far_min < 0) return false;

    *t = t_near_max;
    return true;
}

__device__ Vector3 trace_ray(Ray ray, Sphere* spheres, int num_spheres,
                             Cylinder* cylinders, int num_cylinders,
                             Plane* planes, int num_planes,
                             Rectangle* rectangles, int num_rectangles,
                             Cube* cubes, int num_cubes,
                             Vector3 light_pos, int depth) {
    if (depth > 5) return {0.0f, 0.0f, 0.0f};

    float closest_t = INFINITY;
    Material* closest_material = NULL;
    Vector3 normal;
    Vector3 hit_point;

    // Check sphere intersections
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(ray, spheres[i], &t) && t < closest_t) {
            closest_t = t;
            closest_material = &spheres[i].material;
            hit_point = vector_add(ray.origin, vector_multiply(ray.direction, t));
            normal = vector_normalize(vector_subtract(hit_point, spheres[i].center));
        }
    }

    // Check cylinder intersections
    for (int i = 0; i < num_cylinders; i++) {
        float t;
        if (intersect_cylinder(ray, cylinders[i], &t) && t < closest_t) {
            closest_t = t;
            closest_material = &cylinders[i].material;
            hit_point = vector_add(ray.origin, vector_multiply(ray.direction, t));
            Vector3 cp = vector_subtract(hit_point, cylinders[i].center);
            cp.y = 0;
            normal = vector_normalize(cp);
        }
    }

    // Check plane intersections
    for (int i = 0; i < num_planes; i++) {
        float t;
        if (intersect_plane(ray, planes[i], &t) && t < closest_t) {
            closest_t = t;
            closest_material = &planes[i].material;
            hit_point = vector_add(ray.origin, vector_multiply(ray.direction, t));
            normal = planes[i].normal;
        }
    }

    // Check rectangle intersections
    for (int i = 0; i < num_rectangles; i++) {
        float t;
        if (intersect_rectangle(ray, rectangles[i], &t) && t < closest_t) {
            closest_t = t;
            closest_material = &rectangles[i].material;
            hit_point = vector_add(ray.origin, vector_multiply(ray.direction, t));
            normal = vector_normalize(vector_cross(rectangles[i].u, rectangles[i].v));
        }
    }

    // Check cube intersections
    for (int i = 0; i < num_cubes; i++) {
        float t;
        if (intersect_cube(ray, cubes[i], &t) && t < closest_t) {
            closest_t = t;
            closest_material = &cubes[i].material;
            hit_point = vector_add(ray.origin, vector_multiply(ray.direction, t));
            Vector3 center = vector_multiply(vector_add(cubes[i].min_point, cubes[i].max_point), 0.5f);
            normal = vector_normalize(vector_subtract(hit_point, center));
        }
    }

    if (closest_material == NULL) {
        // Sky color
        float t = 0.5f * (ray.direction.y + 1.0f);
        return vector_add(
            vector_multiply({1.0f, 1.0f, 1.0f}, 1.0f - t),
            vector_multiply({0.5f, 0.7f, 1.0f}, t)
        );
    }

    // Compute lighting
    Vector3 light_dir = vector_normalize(vector_subtract(light_pos, hit_point));
    float diffuse = fmaxf(vector_dot(normal, light_dir), 0.0f);

    Vector3 view_dir = vector_normalize(vector_multiply(ray.direction, -1.0f));
    Vector3 reflect_dir = vector_normalize(vector_subtract(vector_multiply(normal, 2.0f * vector_dot(normal, light_dir)), light_dir));
    float specular = powf(fmaxf(vector_dot(view_dir, reflect_dir), 0.0f), 50.0f) * closest_material->specular;

    Vector3 color = vector_multiply(closest_material->color, diffuse + 0.1f);  // 0.1 for ambient light
    color = vector_add(color, {specular, specular, specular});

    // Compute reflection
    if (closest_material->reflection > 0 && depth < 5) {
        Vector3 reflect_dir = vector_normalize(vector_subtract(ray.direction, vector_multiply(normal, 2.0f * vector_dot(ray.direction, normal))));
        Ray reflect_ray = {hit_point, reflect_dir};
        Vector3 reflection = trace_ray(reflect_ray, spheres, num_spheres, cylinders, num_cylinders, planes, num_planes, rectangles, num_rectangles, cubes, num_cubes, light_pos, depth + 1);
        color = vector_add(
            vector_multiply(color, 1.0f - closest_material->reflection),
            vector_multiply(reflection, closest_material->reflection)
        );
    }

    return color;
}

extern "C" __global__
void ray_trace_kernel(float* output, int width, int height, int samples,
                      float* spheres_data, int num_spheres,
                      float* cylinders_data, int num_cylinders,
                      float* planes_data, int num_planes,
                      float* rectangles_data, int num_rectangles,
                      float* cubes_data, int num_cubes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    curandState state;
    curand_init(y * width + x, 0, 0, &state);

    Sphere spheres[10];
    for (int i = 0; i < num_spheres; i++) {
        spheres[i].center = {spheres_data[i*11], spheres_data[i*11+1], spheres_data[i*11+2]};
        spheres[i].radius = spheres_data[i*11+3];
        spheres[i].material.color = {spheres_data[i*11+4], spheres_data[i*11+5], spheres_data[i*11+6]};
        spheres[i].material.specular = spheres_data[i*11+7];
        spheres[i].material.reflection = spheres_data[i*11+8];
        spheres[i].material.refraction = spheres_data[i*11+9];
        spheres[i].material.refractive_index = spheres_data[i*11+10];
    }

    Cylinder cylinders[10];
    for (int i = 0; i < num_cylinders; i++) {
        cylinders[i].center = {cylinders_data[i*12], cylinders_data[i*12+1], cylinders_data[i*12+2]};
        cylinders[i].radius = cylinders_data[i*12+3];
        cylinders[i].height = cylinders_data[i*12+4];
        cylinders[i].material.color = {cylinders_data[i*12+5], cylinders_data[i*12+6], cylinders_data[i*12+7]};
        cylinders[i].material.specular = cylinders_data[i*12+8];
        cylinders[i].material.reflection = cylinders_data[i*12+9];
        cylinders[i].material.refraction = cylinders_data[i*12+10];
        cylinders[i].material.refractive_index = cylinders_data[i*12+11];
    }

    Plane planes[10];
    for (int i = 0; i < num_planes; i++) {
        planes[i].point = {planes_data[i*11], planes_data[i*11+1], planes_data[i*11+2]};
        planes[i].normal = {planes_data[i*11+3], planes_data[i*11+4], planes_data[i*11+5]};
        planes[i].material.color = {planes_data[i*11+6], planes_data[i*11+7], planes_data[i*11+8]};
        planes[i].material.specular = planes_data[i*11+9];
        planes[i].material.reflection = planes_data[i*11+10];
    }

    Rectangle rectangles[10];
    for (int i = 0; i < num_rectangles; i++) {
        rectangles[i].corner = {rectangles_data[i*14], rectangles_data[i*14+1], rectangles_data[i*14+2]};
        rectangles[i].u = {rectangles_data[i*14+3], rectangles_data[i*14+4], rectangles_data[i*14+5]};
        rectangles[i].v = {rectangles_data[i*14+6], rectangles_data[i*14+7], rectangles_data[i*14+8]};
        rectangles[i].material.color = {rectangles_data[i*14+9], rectangles_data[i*14+10], rectangles_data[i*14+11]};
        rectangles[i].material.specular = rectangles_data[i*14+12];
        rectangles[i].material.reflection = rectangles_data[i*14+13];
    }

    Cube cubes[10];
    for (int i = 0; i < num_cubes; i++) {
        cubes[i].min_point = {cubes_data[i*13], cubes_data[i*13+1], cubes_data[i*13+2]};
        cubes[i].max_point = {cubes_data[i*13+3], cubes_data[i*13+4], cubes_data[i*13+5]};
        cubes[i].material.color = {cubes_data[i*13+6], cubes_data[i*13+7], cubes_data[i*13+8]};
        cubes[i].material.specular = cubes_data[i*13+9];
        cubes[i].material.reflection = cubes_data[i*13+10];
        cubes[i].material.refraction = cubes_data[i*13+11];
        cubes[i].material.refractive_index = cubes_data[i*13+12];
    }

    Vector3 camera_pos = {0.0f, 0.0f, 0.0f};
    Vector3 light_pos = {5.0f, 5.0f, 5.0f};

    Vector3 color = {0.0f, 0.0f, 0.0f};
    for (int s = 0; s < samples; s++) {
        float u = (float(x) + curand_uniform(&state)) / float(width);
        float v = (float(y) + curand_uniform(&state)) / float(height);

        Vector3 direction = vector_normalize({
            (2.0f * u - 1.0f) * float(width) / float(height),
            -(2.0f * v - 1.0f),
            -1.0f
        });

        Ray ray = {camera_pos, direction};
        color = vector_add(color, trace_ray(ray, spheres, num_spheres, cylinders, num_cylinders, planes, num_planes, rectangles, num_rectangles, cubes, num_cubes, light_pos, 0));
    }
    color = vector_multiply(color, 1.0f / float(samples));

    int idx = (y * width + x) * 3;
    output[idx] = fminf(color.x, 1.0f);
    output[idx + 1] = fminf(color.y, 1.0f);
    output[idx + 2] = fminf(color.z, 1.0f);
}

