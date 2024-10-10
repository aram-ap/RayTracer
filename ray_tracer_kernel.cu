// ray_tracer_kernel.cu

#include <curand_kernel.h>
#include <math_constants.h>

struct Ray {
    float3 origin;
    float3 direction;
};

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    float specular;
    float reflection;
};

__device__ float3 vector_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 vector_subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 vector_multiply(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 vector_multiply_vec(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float vector_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 vector_normalize(float3 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / length, v.y / length, v.z / length);
}

__device__ bool intersect_sphere(Ray ray, Sphere sphere, float* t) {
    float3 oc = vector_subtract(ray.origin, sphere.center);
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

__device__ float3 trace_ray(Ray ray, Sphere* spheres, int num_spheres, float3 light_pos, int depth) {
    if (depth > 5) return make_float3(0.0f, 0.0f, 0.0f);

    float closest_t = CUDART_INF_F;
    Sphere* closest_sphere = NULL;

    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(ray, spheres[i], &t) && t < closest_t) {
            closest_t = t;
            closest_sphere = &spheres[i];
        }
    }

    if (closest_sphere == NULL) {
        // Sky color
        float t = 0.5f * (ray.direction.y + 1.0f);
        return vector_add(
            vector_multiply(make_float3(1.0f, 1.0f, 1.0f), 1.0f - t),
            vector_multiply(make_float3(0.5f, 0.7f, 1.0f), t)
        );
    }

    float3 hit_point = vector_add(ray.origin, vector_multiply(ray.direction, closest_t));
    float3 normal = vector_normalize(vector_subtract(hit_point, closest_sphere->center));

    // Diffuse lighting
    float3 light_dir = vector_normalize(vector_subtract(light_pos, hit_point));
    float diffuse = fmaxf(vector_dot(normal, light_dir), 0.0f);

    // Specular lighting
    float3 view_dir = vector_normalize(vector_multiply(ray.direction, -1.0f));
    float3 reflect_dir = vector_normalize(vector_subtract(vector_multiply(normal, 2.0f * vector_dot(normal, light_dir)), light_dir));
    float specular = powf(fmaxf(vector_dot(view_dir, reflect_dir), 0.0f), 50.0f) * closest_sphere->specular;

    float3 color = vector_multiply(closest_sphere->color, diffuse + 0.1f);  // 0.1 for ambient light
    color = vector_add(color, make_float3(specular, specular, specular));

    // Reflection
    if (closest_sphere->reflection > 0 && depth < 5) {
        float3 reflect_dir = vector_normalize(vector_subtract(ray.direction, vector_multiply(normal, 2.0f * vector_dot(ray.direction, normal))));
        Ray reflect_ray = {hit_point, reflect_dir};
        float3 reflection = trace_ray(reflect_ray, spheres, num_spheres, light_pos, depth + 1);
        color = vector_add(
            vector_multiply(color, 1.0f - closest_sphere->reflection),
            vector_multiply(reflection, closest_sphere->reflection)
        );
    }

    return color;
}

extern "C" __global__
void ray_trace_kernel(float* output, int width, int height, int samples,
                      float* spheres_data, int num_spheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    curandState state;
    curand_init(y * width + x, 0, 0, &state);

    Sphere spheres[10];  // Assuming a maximum of 10 spheres
    for (int i = 0; i < num_spheres; i++) {
        spheres[i].center = make_float3(spheres_data[i*10], spheres_data[i*10+1], spheres_data[i*10+2]);
        spheres[i].radius = spheres_data[i*10+3];
        spheres[i].color = make_float3(spheres_data[i*10+4], spheres_data[i*10+5], spheres_data[i*10+6]);
        spheres[i].specular = spheres_data[i*10+7];
        spheres[i].reflection = spheres_data[i*10+8];
    }

    float3 camera_pos = make_float3(0.0f, 0.0f, 0.0f);
    float3 light_pos = make_float3(5.0f, 5.0f, 5.0f);

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples; s++) {
        float u = (float(x) + curand_uniform(&state)) / float(width);
        float v = (float(y) + curand_uniform(&state)) / float(height);

        float3 direction = vector_normalize(make_float3(
            (2.0f * u - 1.0f) * float(width) / float(height),
            -(2.0f * v - 1.0f),
            -1.0f
        ));

        Ray ray = {camera_pos, direction};
        color = vector_add(color, trace_ray(ray, spheres, num_spheres, light_pos, 0));
    }
    color = vector_multiply(color, 1.0f / float(samples));

    int idx = (y * width + x) * 3;
    output[idx] = fminf(color.x, 1.0f);
    output[idx + 1] = fminf(color.y, 1.0f);
    output[idx + 2] = fminf(color.z, 1.0f);
}

