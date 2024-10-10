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


__device__ float3 trace_ray(Ray ray, float* spheres, int num_spheres, float* planes, int num_planes, int depth) {
    if (depth > 5) return make_float3(0, 0, 0);

    float closest_t = 1e30f;
    int closest_obj_index = -1;
    bool is_sphere = true;

    // Check sphere intersections
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(ray, &spheres[i * 11], &t) && t < closest_t) {
            closest_t = t;
            closest_obj_index = i;
            is_sphere = true;
        }
    }

    // Check plane intersections
    for (int i = 0; i < num_planes; i++) {
        float t;
        if (intersect_plane(ray, &planes[i * 11], &t) && t < closest_t) {
            closest_t = t;
            closest_obj_index = i;
            is_sphere = false;
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

    if (is_sphere) {
        float* sphere = &spheres[closest_obj_index * 11];
        normal = vector_normalize(vector_subtract(hit_point, make_float3(sphere[0], sphere[1], sphere[2])));
        material.color = make_float3(sphere[4], sphere[5], sphere[6]);
        material.specular = sphere[7];
        material.reflection = sphere[8];
        material.refraction = sphere[9];
        material.refractive_index = sphere[10];
    } else {
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
        float3 reflection = trace_ray(reflect_ray, spheres, num_spheres, planes, num_planes, depth + 1);
        color = vector_add(vector_multiply(color, 1 - material.reflection), vector_multiply(reflection, material.reflection));
    }

    return color;
}

extern "C" __global__
void ray_trace_kernel(float* output, int width, int height, int samples,
                      float* spheres, int num_spheres,
                      float* planes, int num_planes) {
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
        color = vector_add(color, trace_ray(ray, spheres, num_spheres, planes, num_planes, 0));
    }
    color = vector_multiply(color, 1.0f / float(samples));

    int idx = (y * width + x) * 3;
    output[idx] = fminf(color.x, 1.0f);
    output[idx + 1] = fminf(color.y, 1.0f);
    output[idx + 2] = fminf(color.z, 1.0f);
}