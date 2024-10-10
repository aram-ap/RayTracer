#include <curand_kernel.h>

struct Ray {
    float3 origin;
    float3 direction;
};

struct Material {
    float3 color;
    float specular;
    float reflection;
    float refraction;
    float refractive_index;
};

__device__ float3 vector_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float3 vector_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 vector_subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 vector_multiply(float3 v, float scalar) {
    return make_float3(v.x * scalar, v.y * scalar, v.z * scalar);
}

__device__ float vector_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 vector_normalize(float3 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0) {
        return make_float3(v.x / length, v.y / length, v.z / length);
    }
    return v;
}

__device__ bool is_checkerboard(float3 point) {
    int x = floorf(point.x);
    int z = floorf(point.z);
    return (x + z) % 2 == 0;
}

__device__ bool intersect_sphere(Ray ray, float* sphere, float* t) {
    float3 center = make_float3(sphere[0], sphere[1], sphere[2]);
    float radius = sphere[3];
    float3 oc = vector_subtract(ray.origin, center);
    float a = vector_dot(ray.direction, ray.direction);
    float b = 2.0f * vector_dot(oc, ray.direction);
    float c = vector_dot(oc, oc) - radius * radius;
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

__device__ bool intersect_plane(Ray ray, float* plane, float* t) {
    float3 point = make_float3(plane[0], plane[1], plane[2]);
    float3 normal = make_float3(plane[3], plane[4], plane[5]);
    float denom = vector_dot(normal, ray.direction);
    if (fabsf(denom) > 1e-6) {
        float3 p0l0 = vector_subtract(point, ray.origin);
        *t = vector_dot(p0l0, normal) / denom;
        return (*t >= 0);
    }
    return false;
}

__device__ bool intersect_cylinder(Ray ray, float* cylinder, float* t) {
    float3 center = make_float3(cylinder[0], cylinder[1], cylinder[2]);
    float radius = cylinder[3];
    float height = cylinder[4];

    float3 ro = vector_subtract(ray.origin, center);
    float3 rd = ray.direction;

    float a = rd.x * rd.x + rd.z * rd.z;
    float b = 2 * (ro.x * rd.x + ro.z * rd.z);
    float c = ro.x * ro.x + ro.z * ro.z - radius * radius;

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;

    float t0 = (-b - sqrtf(discriminant)) / (2 * a);
    float t1 = (-b + sqrtf(discriminant)) / (2 * a);

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    float y0 = ro.y + t0 * rd.y;
    float y1 = ro.y + t1 * rd.y;

    if (y0 < 0) {
        if (y1 < 0) return false;
        float th = t0 + (t1 - t0) * (-y0) / (y1 - y0);
        if (th > 0 && th < *t) {
            *t = th;
            return true;
        }
    } else if (y0 >= 0 && y0 <= height) {
        if (t0 > 0 && t0 < *t) {
            *t = t0;
            return true;
        }
    }

    return false;
}

__device__ bool intersect_rectangle(Ray ray, float* rectangle, float* t) {
    float3 corner = make_float3(rectangle[0], rectangle[1], rectangle[2]);
    float3 u = make_float3(rectangle[3], rectangle[4], rectangle[5]);
    float3 v = make_float3(rectangle[6], rectangle[7], rectangle[8]);

    float3 normal = vector_normalize(vector_cross(u, v));
    float d = -vector_dot(normal, corner);

    float denom = vector_dot(normal, ray.direction);
    if (fabsf(denom) < 1e-6) return false;

    *t = -(vector_dot(normal, ray.origin) + d) / denom;
    if (*t < 0) return false;

    float3 p = vector_add(ray.origin, vector_multiply(ray.direction, *t));
    float3 vi = vector_subtract(p, corner);

    float a1 = vector_dot(vi, u);
    if (a1 < 0 || a1 > vector_dot(u, u)) return false;

    float a2 = vector_dot(vi, v);
    if (a2 < 0 || a2 > vector_dot(v, v)) return false;

    return true;
}

__device__ float3 vector_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}


__device__ bool is_in_shadow(float3 hit_point, float3 light_dir, float* spheres, int num_spheres, float* planes, int num_planes) {
    Ray shadow_ray = {hit_point, light_dir};
    float t;
    for (int i = 0; i < num_spheres; i++) {
        if (intersect_sphere(shadow_ray, &spheres[i * 11], &t) && t > 0.001f) {
            return true;
        }
    }
    for (int i = 0; i < num_planes; i++) {
        if (intersect_plane(shadow_ray, &planes[i * 11], &t) && t > 0.001f) {
            return true;
        }
    }
    return false;
}


__device__ float3 trace_ray(Ray ray, float* spheres, int num_spheres, float* cylinders, int num_cylinders, float* planes, int num_planes, float* rectangles, int num_rectangles, int depth) {

    if (depth > 5) return make_float3(0, 0, 0);

    float closest_t = 1e30f;
    int closest_obj_index = -1;
    int obj_type = -1; // 0: sphere, 1: cylinder, 2: plane, 3: rectangle

    // Check sphere intersections
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(ray, &spheres[i * 11], &t) && t < closest_t) {
            closest_t = t;
            closest_obj_index = i;
            obj_type = 0;
        }
    }

    // Check cylinder intersections
    for (int i = 0; i < num_cylinders; i++) {
        float t;
        if (intersect_cylinder(ray, &cylinders[i * 12], &t) && t < closest_t) {
            closest_t = t;
            closest_obj_index = i;
            obj_type = 1;
        }
    }

    // Check plane intersections
    for (int i = 0; i < num_planes; i++) {
        float t;
        if (intersect_plane(ray, &planes[i * 11], &t) && t < closest_t) {
            closest_t = t;
            closest_obj_index = i;
            obj_type = 2;
        }
    }

    // Check rectangle intersections
    for (int i = 0; i < num_rectangles; i++) {
        float t;
        if (intersect_rectangle(ray, &rectangles[i * 14], &t) && t < closest_t) {
            closest_t = t;
            closest_obj_index = i;
            obj_type = 3;
        }
    }

    if (closest_obj_index == -1) {
        // Sky color
        float t = 0.5f * (ray.direction.y + 1.0f);
        return make_float3(1.0f - 0.5f * t, 1.0f - 0.3f * t, 1.0f);
    }

    float3 hit_point = vector_add(ray.origin, vector_multiply(ray.direction, closest_t));
    float3 normal;
    Material material;

    switch (obj_type) {
        case 0: // Sphere
            {
                float* sphere = &spheres[closest_obj_index * 11];
                normal = vector_normalize(vector_subtract(hit_point, make_float3(sphere[0], sphere[1], sphere[2])));
                material.color = make_float3(sphere[4], sphere[5], sphere[6]);
                material.specular = sphere[7];
                material.reflection = sphere[8];
                material.refraction = sphere[9];
                material.refractive_index = sphere[10];
            }
            break;
        case 1: // Cylinder
            {
                float* cylinder = &cylinders[closest_obj_index * 12];
                float3 center = make_float3(cylinder[0], cylinder[1], cylinder[2]);
                normal = vector_normalize(make_float3(hit_point.x - center.x, 0, hit_point.z - center.z));
                material.color = make_float3(cylinder[5], cylinder[6], cylinder[7]);
                material.specular = cylinder[8];
                material.reflection = cylinder[9];
                material.refraction = cylinder[10];
                material.refractive_index = cylinder[11];
            }
            break;
        case 2: // Plane
            {
                float* plane = &planes[closest_obj_index * 11];
                normal = make_float3(plane[3], plane[4], plane[5]);
                if (is_checkerboard(hit_point)) {
                    material.color = make_float3(0.1f, 0.1f, 0.1f);  // Dark color
                } else {
                    material.color = make_float3(0.9f, 0.9f, 0.9f);  // Light color
                }
                material.specular = plane[9];
                material.reflection = plane[10];
                material.refraction = 0;
                material.refractive_index = 1;
            }
            break;
        case 3: // Rectangle
            {
                float* rectangle = &rectangles[closest_obj_index * 14];
                float3 u = make_float3(rectangle[3], rectangle[4], rectangle[5]);
                float3 v = make_float3(rectangle[6], rectangle[7], rectangle[8]);
                normal = vector_normalize(vector_cross(u, v));
                material.color = make_float3(rectangle[9], rectangle[10], rectangle[11]);
                material.specular = rectangle[12];
                material.reflection = rectangle[13];
                material.refraction = 0;
                material.refractive_index = 1;
            }
            break;
    }


    float3 light_pos = make_float3(5, 5, 5);  // Light position
    float3 light_dir = vector_normalize(vector_subtract(light_pos, hit_point));
    float3 view_dir = vector_normalize(vector_multiply(ray.direction, -1.0f));
    float shadow_factor = is_in_shadow(vector_add(hit_point, vector_multiply(normal, 0.001f)), light_dir, spheres, num_spheres, planes, num_planes) ? 0.5f : 1.0f;

    // Ambient
    float3 ambient = vector_multiply(material.color, 0.1f);

    // Diffuse
    float diffuse = fmaxf(vector_dot(normal, light_dir), 0.0f);
    float3 diffuse_color = vector_multiply(material.color, diffuse);

    // Specular
    float3 reflect_dir = vector_normalize(vector_subtract(vector_multiply(normal, 2.0f * vector_dot(normal, light_dir)), light_dir));
    float specular = powf(fmaxf(vector_dot(view_dir, reflect_dir), 0.0f), 32.0f) * material.specular * shadow_factor;
    float3 specular_color = make_float3(specular, specular, specular);

    float3 color = vector_add(vector_add(ambient, diffuse_color), specular_color);

    if (material.reflection > 0 && depth < 5) {
        float3 reflect_dir = vector_subtract(ray.direction, vector_multiply(normal, 2 * vector_dot(ray.direction, normal)));
        Ray reflect_ray = {hit_point, reflect_dir};
        float3 reflection = trace_ray(reflect_ray, spheres, num_spheres, cylinders, num_cylinders, planes, num_planes, rectangles, num_rectangles, depth + 1);
        color = vector_add(vector_multiply(color, 1 - material.reflection), vector_multiply(reflection, material.reflection));
    }

    return color;
}

extern "C" __global__
void ray_trace_kernel(float* output, int width, int height, int samples,
                      float* spheres, int num_spheres,
                      float* cylinders, int num_cylinders,
                      float* planes, int num_planes,
                      float* rectangles, int num_rectangles) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    curandState state;
    curand_init(y * width + x, 0, 0, &state);

    float3 color = make_float3(0, 0, 0);
    for (int s = 0; s < samples; s++) {
        float u = (float(x) + curand_uniform(&state)) / float(width);
        float v = (float(y) + curand_uniform(&state)) / float(height);

        float3 direction = vector_normalize(make_float3(
            (2.0f * u - 1.0f) * float(width) / float(height),
            -(2.0f * v - 1.0f),
            -1.0f
        ));

        Ray ray = {make_float3(0, 0, 0), direction};
        color = vector_add(color, trace_ray(ray, spheres, num_spheres, cylinders, num_cylinders, planes, num_planes, rectangles, num_rectangles, 0));
    }
    color = vector_multiply(color, 1.0f / float(samples));

    int idx = (y * width + x) * 3;
    output[idx] = fminf(color.x, 1.0f);
    output[idx + 1] = fminf(color.y, 1.0f);
    output[idx + 2] = fminf(color.z, 1.0f);
}
