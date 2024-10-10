#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_DEPTH 5
#define EPSILON 0.001f


#include <curand_kernel.h>

__device__ float3 vector_subtract(float3 a, float3 b);
__device__ float3 vector_add(float3 a, float3 b);
__device__ float3 vector_multiply(float3 a, float3 b);
__device__ float3 vector_multiply(float3 v, float scalar);
__device__ float vector_dot(float3 a, float3 b);
__device__ float3 vector_cross(float3 a, float3 b);
__device__ float3 vector_normalize(float3 v);


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
    float metallic;
};

__device__ float3 reflect(float3 incident, float3 normal) {
    return vector_subtract(incident, vector_multiply(normal, 2.0f * vector_dot(incident, normal)));
}

__device__ float3 vector_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float3 vector_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 vector_subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 vector_multiply(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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

__device__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ bool intersect_sphere(float3 ray_origin, float3 ray_direction, float* sphere, float* t) {
    float3 center = make_float3(sphere[0], sphere[1], sphere[2]);
    float radius = sphere[3];

    float3 oc = vector_subtract(ray_origin, center);
    float a = vector_dot(ray_direction, ray_direction);
    float b = 2.0f * vector_dot(oc, ray_direction);
    float c = vector_dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0f * a);
    float t1 = (-b + sqrt_disc) / (2.0f * a);

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    if (t0 < 0) {
        t0 = t1;
        if (t0 < 0) return false;
    }

    *t = t0;
    return true;
}

__device__ bool intersect_cylinder(float3 ray_origin, float3 ray_direction, float* cylinder, float* t) {
    float3 center = make_float3(cylinder[0], cylinder[1], cylinder[2]);
    float radius = cylinder[3];
    float height = cylinder[4];

    // Assume cylinder is aligned with y-axis
    float a = ray_direction.x * ray_direction.x + ray_direction.z * ray_direction.z;
    float b = 2 * (ray_direction.x * (ray_origin.x - center.x) + ray_direction.z * (ray_origin.z - center.z));
    float c = (ray_origin.x - center.x) * (ray_origin.x - center.x) +
              (ray_origin.z - center.z) * (ray_origin.z - center.z) - radius * radius;

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;

    float sqrt_disc = sqrtf(discriminant);
    float t0 = (-b - sqrt_disc) / (2.0f * a);
    float t1 = (-b + sqrt_disc) / (2.0f * a);

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    float y0 = ray_origin.y + t0 * ray_direction.y;
    float y1 = ray_origin.y + t1 * ray_direction.y;

    float y_min = center.y;
    float y_max = center.y + height;

    if (y0 < y_min) {
        if (y1 < y_min) return false;
        float t_y = (y_min - ray_origin.y) / ray_direction.y;
        if (t_y > t1) return false;
        if (t_y > t0) t0 = t_y;
    } else if (y0 > y_max) {
        if (y1 > y_max) return false;
        float t_y = (y_max - ray_origin.y) / ray_direction.y;
        if (t_y < t0) return false;
        if (t_y < t1) t1 = t_y;
    }

    *t = t0;
    return true;
}

__device__ bool intersect_plane(float3 ray_origin, float3 ray_direction, float* plane, float* t) {
    float3 normal = make_float3(plane[0], plane[1], plane[2]);
    float d = plane[3];

    float denom = vector_dot(normal, ray_direction);
    if (fabsf(denom) < 1e-6f) return false;

    *t = -(vector_dot(normal, ray_origin) + d) / denom;
    return *t >= 0;
}

__device__ bool intersect_rectangle(float3 ray_origin, float3 ray_direction, float* rectangle, float* t) {
    float3 corner = make_float3(rectangle[0], rectangle[1], rectangle[2]);
    float3 edge1 = make_float3(rectangle[3], rectangle[4], rectangle[5]);
    float3 edge2 = make_float3(rectangle[6], rectangle[7], rectangle[8]);

    float3 normal = vector_normalize(vector_cross(edge1, edge2));
    float d = -vector_dot(normal, corner);

    float denom = vector_dot(normal, ray_direction);
    if (fabsf(denom) < 1e-6f) return false;

    *t = -(vector_dot(normal, ray_origin) + d) / denom;
    if (*t < 0) return false;

    float3 hit_point = vector_add(ray_origin, vector_multiply(ray_direction, *t));
    float3 v = vector_subtract(hit_point, corner);

    float dot11 = vector_dot(edge1, edge1);
    float dot12 = vector_dot(edge1, edge2);
    float dot22 = vector_dot(edge2, edge2);
    float dot1v = vector_dot(edge1, v);
    float dot2v = vector_dot(edge2, v);

    float invDenom = 1.0f / (dot11 * dot22 - dot12 * dot12);
    float u = (dot22 * dot1v - dot12 * dot2v) * invDenom;
    float cv = (dot11 * dot2v - dot12 * dot1v) * invDenom;

    return (u >= 0 && u <= 1 && cv >= 0 && cv <= 1);
}

__device__ bool intersect_cube(float3 ray_origin, float3 ray_direction, float* cube, float* t) {
    float3 min_point = make_float3(cube[0], cube[1], cube[2]);
    float3 max_point = make_float3(cube[3], cube[4], cube[5]);

    float3 inv_dir = make_float3(1.0f / ray_direction.x, 1.0f / ray_direction.y, 1.0f / ray_direction.z);

    float3 t1 = vector_multiply(vector_subtract(min_point, ray_origin), inv_dir);
    float3 t2 = vector_multiply(vector_subtract(max_point, ray_origin), inv_dir);

    float3 tmin = make_float3(fminf(t1.x, t2.x), fminf(t1.y, t2.y), fminf(t1.z, t2.z));
    float3 tmax = make_float3(fmaxf(t1.x, t2.x), fmaxf(t1.y, t2.y), fmaxf(t1.z, t2.z));

    float tnear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float tfar = fminf(fminf(tmax.x, tmax.y), tmax.z);

    if (tnear > tfar || tfar < 0) return false;

    *t = tnear;
    return true;
}


__device__ bool is_checkerboard(float3 point, float scale = 1.0f) {
    int x = floorf(point.x * scale);
    int z = floorf(point.z * scale);
    return (x + z) % 2 == 0;
}

__device__ bool is_in_shadow(Ray shadow_ray, float* spheres, int num_spheres,
                             float* cylinders, int num_cylinders,
                             float* planes, int num_planes,
                             float* rectangles, int num_rectangles,
                             float* cubes, int num_cubes) {
    float t;
    float light_distance = length(shadow_ray.direction);
    float3 light_dir = vector_normalize(shadow_ray.direction);

    // Check spheres
    for (int i = 0; i < num_spheres; i++) {
        if (intersect_sphere(shadow_ray.origin, light_dir, &spheres[i * 11], &t)) {
            if (t > EPSILON && t < light_distance) return true;
        }
    }

    // Check cylinders
    for (int i = 0; i < num_cylinders; i++) {
        if (intersect_cylinder(shadow_ray.origin, light_dir, &cylinders[i * 12], &t)) {
            if (t > EPSILON && t < light_distance) return true;
        }
    }

    // Check planes
    for (int i = 0; i < num_planes; i++) {
        if (intersect_plane(shadow_ray.origin, light_dir, &planes[i * 11], &t)) {
            if (t > EPSILON && t < light_distance) return true;
        }
    }

    // Check rectangles
    for (int i = 0; i < num_rectangles; i++) {
        if (intersect_rectangle(shadow_ray.origin, light_dir, &rectangles[i * 14], &t)) {
            if (t > EPSILON && t < light_distance) return true;
        }
    }

    // Check cubes
    for (int i = 0; i < num_cubes; i++) {
        if (intersect_cube(shadow_ray.origin, light_dir, &cubes[i * 13], &t)) {
            if (t > EPSILON && t < light_distance) return true;
        }
    }

    return false;
}

__device__ Ray create_camera_ray(float u, float v, float3 camera_position, float3 camera_look_at, float3 camera_up, float aspect_ratio, float fov) {
    // Calculate camera basis vectors
    float3 w = vector_normalize(vector_subtract(camera_position, camera_look_at));
    float3 u_vec = vector_normalize(vector_cross(camera_up, w));
    float3 v_vec = vector_cross(w, u_vec);

    // Calculate viewport dimensions
    float theta = fov * M_PI / 180.0f;
    float half_height = tanf(theta / 2.0f);
    float half_width = aspect_ratio * half_height;

    // Calculate pixel position on viewport
    float3 viewport_u = vector_multiply(u_vec, 2.0f * half_width);
    float3 viewport_v = vector_multiply(v_vec, 2.0f * half_height);
    float3 viewport_lower_left = vector_subtract(
        vector_subtract(
            vector_subtract(camera_position, vector_multiply(w, 1.0f)),
            vector_multiply(viewport_u, 0.5f)
        ),
        vector_multiply(viewport_v, 0.5f)
    );

    // Calculate ray direction
    float3 direction = vector_add(
        viewport_lower_left,
        vector_add(
            vector_multiply(viewport_u, u),
            vector_multiply(viewport_v, v)
        )
    );
    direction = vector_subtract(direction, camera_position);
    direction = vector_normalize(direction);

    // Create and return the ray
    Ray ray;
    ray.origin = camera_position;
    ray.direction = direction;
    return ray;
}


__device__ float3 trace_ray(Ray ray, float3 light_pos,
                            float* spheres, int num_spheres,
                            float* cylinders, int num_cylinders,
                            float* planes, int num_planes,
                            float* rectangles, int num_rectangles,
                            float* cubes, int num_cubes,
                            int depth) {
    if (depth > MAX_DEPTH) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float closest_t = 1e30f;
    int closest_obj_index = -1;
    int obj_type = -1; // 0: sphere, 1: cylinder, 2: plane, 3: rectangle, 4: cube

    // Check spheres
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersect_sphere(ray.origin, ray.direction, &spheres[i * 11], &t)) {
            if (t > 0 && t < closest_t) {
                closest_t = t;
                closest_obj_index = i;
                obj_type = 0;
            }
        }
    }

    // Check cylinders
    for (int i = 0; i < num_cylinders; i++) {
        float t;
        if (intersect_cylinder(ray.origin, ray.direction, &cylinders[i * 12], &t)) {
            if (t > 0 && t < closest_t) {
                closest_t = t;
                closest_obj_index = i;
                obj_type = 1;
            }
        }
    }

    // Check planes
    for (int i = 0; i < num_planes; i++) {
        float t;
        if (intersect_plane(ray.origin, ray.direction, &planes[i * 11], &t)) {
            if (t > 0 && t < closest_t) {
                closest_t = t;
                closest_obj_index = i;
                obj_type = 2;
            }
        }
    }

    // Check rectangles
    for (int i = 0; i < num_rectangles; i++) {
        float t;
        if (intersect_rectangle(ray.origin, ray.direction, &rectangles[i * 14], &t)) {
            if (t > 0 && t < closest_t) {
                closest_t = t;
                closest_obj_index = i;
                obj_type = 3;
            }
        }
    }

    // Check cubes
    for (int i = 0; i < num_cubes; i++) {
        float t;
        if (intersect_cube(ray.origin, ray.direction, &cubes[i * 13], &t)) {
            if (t > 0 && t < closest_t) {
                closest_t = t;
                closest_obj_index = i;
                obj_type = 4;
            }
        }
    }

    if (closest_obj_index == -1) {
        // No intersection, return background color
        float t = 0.5f * (ray.direction.y + 1.0f);
        return make_float3(1.0f - 0.5f * t, 1.0f - 0.3f * t, 1.0f);
    }

    // Calculate hit point
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
//                normal = closest_normal;
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
            if (is_checkerboard(hit_point, 0.5f)) {
                material.color = make_float3(0.8f, 0.8f, 0.8f);  // Light gray
            } else {
                material.color = make_float3(0.5f, 0.5f, 0.5f);  // Medium gray
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
        case 4: // Cube
            {
                float* cube = &cubes[closest_obj_index * 13];
                float3 min_point = make_float3(cube[0], cube[1], cube[2]);
                float3 max_point = make_float3(cube[3], cube[4], cube[5]);
                normal = make_float3(
                    (hit_point.x - min_point.x < 0.001f) ? -1.0f : ((hit_point.x - max_point.x > -0.001f) ? 1.0f : 0.0f),
                    (hit_point.y - min_point.y < 0.001f) ? -1.0f : ((hit_point.y - max_point.y > -0.001f) ? 1.0f : 0.0f),
                    (hit_point.z - min_point.z < 0.001f) ? -1.0f : ((hit_point.z - max_point.z > -0.001f) ? 1.0f : 0.0f)
                );
                normal = vector_normalize(normal);
                material.color = make_float3(cube[6], cube[7], cube[8]);
                material.specular = cube[9];
                material.reflection = cube[10];
                material.refraction = cube[11];
                material.refractive_index = cube[12];
            }
            break;
    }

//    float3 light_pos = make_float3(5, 5, 5);  // Light position
    float3 light_dir = vector_normalize(vector_subtract(light_pos, hit_point));
    float3 view_dir = vector_normalize(vector_multiply(ray.direction, -1.0f));

    // Ambient
    float3 ambient = vector_multiply(material.color, 0.1f);

    // Diffuse
    float diffuse_strength = fmaxf(vector_dot(normal, light_dir), 0.0f);
    float3 diffuse = vector_multiply(material.color, diffuse_strength);

    // Specular
    float3 reflect_dir = reflect(vector_multiply(light_dir, -1.0f), normal);
    float specular_strength = powf(fmaxf(vector_dot(view_dir, reflect_dir), 0.0f), 32.0f);
    float3 specular = vector_multiply(make_float3(1.0f, 1.0f, 1.0f), specular_strength * material.specular);

    // Shadow
    float shadow_factor = 1.0f;
    Ray shadow_ray = {hit_point, light_dir};
    shadow_ray.origin = vector_add(shadow_ray.origin, vector_multiply(normal, EPSILON));
    if (is_in_shadow(shadow_ray, spheres, num_spheres, cylinders, num_cylinders,
                     planes, num_planes, rectangles, num_rectangles, cubes, num_cubes)) {
        shadow_factor = 0.1f;
    }

    // Combine lighting
    float3 lighting = vector_add(ambient, vector_multiply(vector_add(diffuse, specular), shadow_factor));

    // Reflection
    float3 reflection_color = make_float3(0.0f, 0.0f, 0.0f);
    if (material.reflection > 0.0f && depth < MAX_DEPTH) {
        Ray reflection_ray = {hit_point, reflect(ray.direction, normal)};
        reflection_ray.origin = vector_add(reflection_ray.origin, vector_multiply(normal, EPSILON));
        reflection_color = trace_ray(reflection_ray, light_pos, spheres, num_spheres, cylinders, num_cylinders,
                planes, num_planes, rectangles, num_rectangles, cubes, num_cubes, depth + 1);
    }

    // Metallic reflection
    if (material.metallic > 0.0f) {
        float3 metallic_color = vector_multiply(reflection_color, material.metallic);
        lighting = vector_add(vector_multiply(lighting, 1.0f - material.metallic), metallic_color);
    }

    // Refraction
    float3 refraction_color = make_float3(0.0f, 0.0f, 0.0f);
    if (material.refraction > 0.0f && depth < MAX_DEPTH) {
        float n1 = 1.0f; // Air refractive index
        float n2 = material.refractive_index;
        float3 n = normal;
        float cos_i = -vector_dot(ray.direction, normal);

        if (cos_i < 0.0f) {
            // Ray is inside the object
            cos_i = -cos_i;
            n = vector_multiply(normal, -1.0f);
            float temp = n1;
            n1 = n2;
            n2 = temp;
        }

        float eta = n1 / n2;
        float k = 1.0f - eta * eta * (1.0f - cos_i * cos_i);

        if (k >= 0.0f) {
            float3 refraction_dir = vector_normalize(vector_add(vector_multiply(ray.direction, eta),
                                                     vector_multiply(n, eta * cos_i - sqrtf(k))));
            Ray refraction_ray = {hit_point, refraction_dir};
            refraction_ray.origin = vector_subtract(refraction_ray.origin, vector_multiply(normal, EPSILON));
            refraction_color = trace_ray(refraction_ray, light_pos, spheres, num_spheres, cylinders, num_cylinders,
                                         planes, num_planes, rectangles, num_rectangles, cubes, num_cubes, depth + 1);
        } else {
            // Total internal reflection
            refraction_color = reflection_color;
        }
    }

    // Combine colors
    float3 final_color = vector_add(
        vector_multiply(lighting, 1.0f - material.reflection - material.refraction),
        vector_add(
            vector_multiply(reflection_color, material.reflection),
            vector_multiply(refraction_color, material.refraction)
        )
    );

    return final_color;
}

extern "C" __global__
void ray_trace_kernel(float* output, int width, int height, int samples,
                      float* spheres, int num_spheres,
                      float* cylinders, int num_cylinders,
                      float* planes, int num_planes,
                      float* rectangles, int num_rectangles,
                      float* cubes, int num_cubes,
                      double* camera_params, float* light) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    curandState state;
    curand_init(y * width + x, 0, 0, &state);

    // Unpack camera parameters
    float3 camera_position = make_float3(camera_params[0], camera_params[1], camera_params[2]);
    float3 camera_look_at = make_float3(camera_params[3], camera_params[4], camera_params[5]);
    float fov = camera_params[6];

    float3 camera_forward = vector_normalize(vector_subtract(camera_look_at, camera_position));
    float3 camera_right = vector_normalize(vector_cross(camera_forward, make_float3(0.0f, 1.0f, 0.0f)));
    float3 camera_up_adjusted = vector_cross(camera_right, camera_forward);

    float aspect_ratio = float(width) / float(height);
    float3 color = make_float3(0, 0, 0);

    for (int s = 0; s < samples; s++) {
        float u = (float(x) + curand_uniform(&state)) / float(width);
        float v = (float(y) + curand_uniform(&state)) / float(height);

        float3 direction = vector_normalize(
            vector_add(
                vector_add(
                    vector_multiply(camera_right, (2.0f * u - 1.0f) * tanf(fov * 0.5f * CUDART_PI_F / 180.0f) * aspect_ratio),
                    vector_multiply(camera_up_adjusted, (1.0f - 2.0f * v) * tanf(fov * 0.5f * CUDART_PI_F / 180.0f))
                ),
                camera_forward
            )
        );

        Ray ray = {camera_position, direction};
        color = vector_add(color, trace_ray(ray, make_float3(light[0], light[1], light[2]), spheres, num_spheres, 0));
    }

    color = vector_multiply(color, 1.0f / float(samples));

    int idx = (y * width + x) * 3;
    output[idx] = fminf(color.x, 1.0f);
    output[idx + 1] = fminf(color.y, 1.0f);
    output[idx + 2] = fminf(color.z, 1.0f);
}

//
//
//extern "C" __global__
//void ray_trace_kernel(float* output, int width, int height, int samples,
//                      float* spheres, int num_spheres,
//                      float* cylinders, int num_cylinders,
//                      float* planes, int num_planes,
//                      float* rectangles, int num_rectangles,
//                      float* cubes, int num_cubes,
//                      float* camera, float* light) {
//
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x >= width || y >= height) return;
//
//    curandState state;
//    curand_init(y * width + x, 0, 0, &state);
//
//    float3 camera_position = make_float3(camera[0], camera[1], camera[2]);
//    float3 camera_look_at = make_float3(camera[3], camera[4], camera[5]);
//    float3 camera_up = make_float3(camera[6], camera[7], camera[8]);
//    float fov = camera[9];
//
//    float3 camera_forward = vector_normalize(vector_subtract(camera_look_at, camera_position));
//    float3 camera_right = vector_normalize(vector_cross(camera_forward, camera_up));
//    float3 camera_up_adjusted = vector_cross(camera_right, camera_forward);
//
//    float aspect_ratio = float(width) / float(height);
//    float3 light_pos = make_float3(light[0], light[1], light[2]);
//
//    float3 color = make_float3(0, 0, 0);
//    for (int s = 0; s < samples; s++) {
//        float u = (float(x) + curand_uniform(&state)) / float(width);
//        float v = (float(y) + curand_uniform(&state)) / float(height);
//
//        float3 direction = vector_normalize(
//            vector_add(
//                vector_add(
//                    vector_multiply(camera_right, (2.0f * u - 1.0f) * tanf(fov * 0.5f * M_PI / 180.0f) * aspect_ratio),
//                    vector_multiply(camera_up_adjusted, (1.0f - 2.0f * v) * tanf(fov * 0.5f * M_PI / 180.0f))
//                ),
//                camera_forward
//            )
//        );
//
//        Ray ray = {camera_position, direction};
//        color = vector_add(color, trace_ray(ray, light_pos, spheres, num_spheres, cylinders, num_cylinders,
//                                            planes, num_planes, rectangles, num_rectangles, cubes, num_cubes, 0));
//    }
//    color = vector_multiply(color, 1.0f / float(samples));
//
//    int idx = (y * width + x) * 3;
//    output[idx] = fminf(color.x, 1.0f);
//    output[idx + 1] = fminf(color.y, 1.0f);
//    output[idx + 2] = fminf(color.z, 1.0f);
//}