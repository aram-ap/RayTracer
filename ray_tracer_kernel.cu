// ray_tracer_kernel.cu
#include <curand_kernel.h>
#include <math_constants.h>

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

struct Sphere {
    float3 center;
    float radius;
    Material material;
};

struct Cylinder {
    float3 center;
    float radius;
    float height;
    Material material;
};

struct Plane {
    float3 point;
    float3 normal;
    Material material;
};

// ... (keep all vector operations as they were) ...

__device__ bool intersect_sphere(Ray ray, Sphere sphere, float* t) {
    // ... (keep the sphere intersection code as it was) ...
}

__device__ bool intersect_cylinder(Ray ray, Cylinder cylinder, float* t) {
    // Implement cylinder intersection
    // This is a simplified version, you might want to improve it
    float3 oc = vector_subtract(ray.origin, cylinder.center);
    float a = ray.direction.x * ray.direction.x + ray.direction.z * ray.direction.z;
    float b = 2.0f * (oc.x * ray.direction.x + oc.z * ray.direction.z);
    float c = oc.x * oc.x + oc.z * oc.z - cylinder.radius * cylinder.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    float t0 = (-b - sqrtf(discriminant)) / (2.0f * a);
    float t1 = (-b + sqrtf(discriminant)) / (2.0f * a);

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    float y0 = ray.origin.y + t0 * ray.direction.y;
    float y1 = ray.origin.y + t1 * ray.direction.y;

    float cylinder_min = cylinder.center.y - cylinder.height / 2.0f;
    float cylinder_max = cylinder.center.y + cylinder.height / 2.0f;

    if (y0 < cylinder_min) {
        if (y1 < cylinder_min) return false;
        t0 = t0 + (t1 - t0) * (cylinder_min - y0) / (y1 - y0);
    } else if (y0 > cylinder_max) {
        if (y1 > cylinder_max) return false;
        t0 = t0 + (t1 - t0) * (cylinder_max - y0) / (y1 - y0);
    }

    *t = t0;
    return true;
}

__device__ bool intersect_plane(Ray ray, Plane plane, float* t) {
    float denom = vector_dot(plane.normal, ray.direction);
    if (fabsf(denom) > 1e-6) {
        float3 p0l0 = vector_subtract(plane.point, ray.origin);
        *t = vector_dot(p0l0, plane.normal) / denom;
        return (*t >= 0);
    }
    return false;
}

__device__ float3 trace_ray(Ray ray, Sphere* spheres, int num_spheres,
                            Cylinder* cylinders, int num_cylinders,
                            Plane* planes, int num_planes,
                            float3 light_pos, int depth) {
    if (depth > 5) return make_float3(0.0f, 0.0f, 0.0f);

    float closest_t = CUDART_INF_F;
    Material* closest_material = NULL;
    float3 normal;
    float3 hit_point;

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
            float3 cp = vector_subtract(hit_point, cylinders[i].center);
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

    if (closest_material == NULL) {
        // Sky color
        float t = 0.5f * (ray.direction.y + 1.0f);
        return vector_add(
            vector_multiply(make_float3(1.0f, 1.0f, 1.0f), 1.0f - t),
            vector_multiply(make_float3(0.5f, 0.7f, 1.0f), t)
        );
    }

    // Compute lighting
    float3 light_dir = vector_normalize(vector_subtract(light_pos, hit_point));
    float diffuse = fmaxf(vector_dot(normal, light_dir), 0.0f);

    float3 view_dir = vector_normalize(vector_multiply(ray.direction, -1.0f));
    float3 reflect_dir = vector_normalize(vector_subtract(vector_multiply(normal, 2.0f * vector_dot(normal, light_dir)), light_dir));
    float specular = powf(fmaxf(vector_dot(view_dir, reflect_dir), 0.0f), 50.0f) * closest_material->specular;

    float3 color = vector_multiply(closest_material->color, diffuse + 0.1f);  // 0.1 for ambient light
    color = vector_add(color, make_float3(specular, specular, specular));

    // Compute reflection
    if (closest_material->reflection > 0 && depth < 5) {
        float3 reflect_dir = vector_normalize(vector_subtract(ray.direction, vector_multiply(normal, 2.0f * vector_dot(ray.direction, normal))));
        Ray reflect_ray = {hit_point, reflect_dir};
        float3 reflection = trace_ray(reflect_ray, spheres, num_spheres, cylinders, num_cylinders, planes, num_planes, light_pos, depth + 1);
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
                      float* planes_data, int num_planes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    curandState state;
    curand_init(y * width + x, 0, 0, &state);

    Sphere spheres[10];
    for (int i = 0; i < num_spheres; i++) {
        spheres[i].center = make_float3(spheres_data[i*11], spheres_data[i*11+1], spheres_data[i*11+2]);
        spheres[i].radius = spheres_data[i*11+3];
        spheres[i].material.color = make_float3(spheres_data[i*11+4], spheres_data[i*11+5], spheres_data[i*11+6]);
        spheres[i].material.specular = spheres_data[i*11+7];
        spheres[i].material.reflection = spheres_data[i*11+8];
        spheres[i].material.refraction = spheres_data[i*11+9];
        spheres[i].material.refractive_index = spheres_data[i*11+10];
    }

    Cylinder cylinders[10];
    for (int i = 0; i < num_cylinders; i++) {
        cylinders[i].center = make_float3(cylinders_data[i*12], cylinders_data[i*12+1], cylinders_data[i*12+2]);
        cylinders[i].radius = cylinders_data[i*12+3];
        cylinders[i].height = cylinders_data[i*12+4];
        cylinders[i].material.color = make_float3(cylinders_data[i*12+5], cylinders_data[i*12+6], cylinders_data[i*12+7]);
        cylinders[i].material.specular = cylinders_data[i*12+8];
        cylinders[i].material.reflection = cylinders_data[i*12+9];
        cylinders[i].material.refraction = cylinders_data[i*12+10];
        cylinders[i].material.refractive_index = cylinders_data[i*12+11];
    }

    Plane planes[10];
    for (int i = 0; i < num_planes; i++) {
        planes[i].point = make_float3(planes_data[i*11], planes_data[i*11+1], planes_data[i*11+2]);
        planes[i].normal = make_float3(planes_data[i*11+3], planes_data[i*11+4], planes_data[i*11+5]);
        planes[i].material.color = make_float3(planes_data[i*11+6], planes_data[i*11+7], planes_data[i*11+8]);
        planes[i].material.specular = planes_data[i*11+9];
        planes[i].material.reflection = planes_data[i*11+10];
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
        color = vector_add(color, trace_ray(ray, spheres, num_spheres, cylinders, num_cylinders, planes, num_planes, light_pos, 0));
    }
    color = vector_multiply(color, 1.0f / float(samples));

    int idx = (y * width + x) * 3;
    output[idx] = fminf(color.x, 1.0f);
    output[idx + 1] = fminf(color.y, 1.0f);
    output[idx + 2] = fminf(color.z, 1.0f);
}

