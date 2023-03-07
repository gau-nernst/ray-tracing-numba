import time

import numba as nb
import numpy as np
from PIL import Image

import rt


DTYPE = np.float64
WIDTH = 1920
HEIGHT = 1080
BACKGROUND_COLOR1 = np.array([0.7, 0.8, 0.9], dtype=DTYPE)
BACKGROUND_COLOR2 = np.array([0.05, 0.05, 0.2], dtype=DTYPE)
ASPECT_RATIO = HEIGHT / WIDTH
WINDOW_DEPTH = 2.0


@nb.njit
def background_color(ray):
    u = (ray[1] + 1) * 0.5  # [-1,1] to [0,1]
    return u * BACKGROUND_COLOR1 + (1.0 - u) * BACKGROUND_COLOR2


@nb.njit
def to_ray_vector(x_ind, y_ind):
    x = (x_ind + 0.5) / WIDTH
    y = (y_ind + 0.5) / HEIGHT
    ray_direction = np.array(
        [x * 2 - 1, (y * 2 - 1) * ASPECT_RATIO, -WINDOW_DEPTH],  # [0,1] to [-1,1]
        dtype=DTYPE,
    )
    return rt.normalize(ray_direction)


@nb.njit
def ray_color(ray_direction, spheres, depth):
    ray_origin = np.zeros(3, dtype=DTYPE)
    pixel = np.ones(3, dtype=DTYPE)

    for _ in range(depth):
        hit_something, t, i = rt.hit_many_spheres(ray_origin, ray_direction, spheres)
        sphere = spheres[i]

        if not hit_something:  # hit background
            pixel *= background_color(ray_direction)
            break

        if hit_something and sphere.material == rt.LIGHT_SOURCE:
            pixel *= sphere.color
            break

        # scatter light
        ray_origin += ray_direction * t
        surface_normal = (ray_origin - sphere.center) / sphere.radius
        ray_direction = rt.reflect(ray_direction, surface_normal)
        pixel *= sphere.color

    return pixel


@nb.njit(parallel=True)
def program(img, spheres):
    img_h, img_w = img.shape[:2]

    n_samples = 4
    max_depth = 50
    for row_idx in range(img_h):
        if row_idx % 50 == 0:
            print("row =", row_idx)

        for col_idx in nb.prange(img_w):
            pixel = img[HEIGHT - 1 - row_idx, col_idx]
            for _ in range(n_samples):
                ray_direction = to_ray_vector(
                    col_idx + np.random.rand(),
                    row_idx + np.random.rand(),
                )
                pixel += ray_color(ray_direction, spheres, max_depth)
            pixel /= n_samples


spheres = nb.typed.List()
spheres.append(
    rt.Sphere(
        np.array([0.0, 3.0, -10.0]),
        1.0,
        np.array([5.0, 5.0, 5.0]),
        rt.LIGHT_SOURCE,
    )
)
big_radius = 10000.0
spheres.append(
    rt.Sphere(
        np.array([0.0, -big_radius - 1, 0.0]),
        big_radius,
        np.array([0.7, 0.7, 0.7]),
        rt.METAL,
    )
)
spheres.append(
    rt.Sphere(
        np.array([1.0, 1.0, -7.0]),
        1.0,
        np.array([0.7, 0.6, 0.5]),
        rt.METAL,
    )
)
spheres.append(
    rt.Sphere(
        np.array([-1.0, 0.0, -6.0]),
        1.0,
        np.array([1.0, 1.0, 1.0]),
        rt.METAL,
    )
)

img = np.zeros((HEIGHT, WIDTH, 3))
time0 = time.perf_counter()
program(img, spheres)
print(time.perf_counter() - time0)

img = (img * img * 255.999).astype(np.uint8)
Image.fromarray(img).save("image.png")
