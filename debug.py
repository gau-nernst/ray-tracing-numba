import time

import numba as nb
import numpy as np
from PIL import Image

import utils

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
    return utils.normalize(ray_direction)


@nb.njit(parallel=True)
def program1(im):
    ray_origin = np.zeros(3)
    big_radius = 10000.0
    radii = np.array([1.0, big_radius, 1.0, 1.0])

    centers = np.array(
        [
            [0.0, 3.0, -10.0],
            [0.0, -big_radius - 1, 0.0],
            [1.0, 1.0, -7.0],
            [-1.0, 0.0, -6.0],
        ],
        dtype=DTYPE,
    )

    # white > 1. although its direct value is clipped to 1, its attenuated reflections will be brighter
    # black, will not emit any color on its own, just reflect color
    emitted_colors = np.array(
        [
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )

    specular_reflectivity = np.array(
        [
            [0.0, 0.0, 0.0],  # does not reflect light
            [0.7, 0.7, 0.7],  # reflect all lights equally
            [0.7, 0.6, 0.5],  # reflect more red than other colors
            [1.0, 1.0, 1.0],  # full reflection
        ],
        dtype=DTYPE,
    )

    for row_idx in range(HEIGHT):
        if row_idx % 50 == 0:
            print("row =", row_idx)

        for col_idx in nb.prange(WIDTH):
            ray_origin = np.zeros(3, dtype=DTYPE)
            multiplier = np.ones(3, dtype=DTYPE)
            ray_direction = to_ray_vector(col_idx, row_idx)

            while True:
                hit_something, t, i = utils.hit_many_spheres(
                    ray_origin, ray_direction, centers, radii
                )

                if not hit_something:  # hit background
                    im[HEIGHT - 1 - row_idx, col_idx, :] = (
                        background_color(ray_direction) * multiplier
                    )
                    break

                if (
                    hit_something and emitted_colors[i].dot(emitted_colors[i]) > 0
                ):  # hit light-emitting object
                    im[HEIGHT - 1 - row_idx, col_idx, :] = (
                        emitted_colors[i] * multiplier
                    )
                    break

                # scatter light
                ray_origin += ray_direction * t
                surface_normal = (ray_origin - centers[i]) / radii[i]
                ray_direction = utils.reflect(ray_direction, surface_normal)
                multiplier *= specular_reflectivity[i]


im = np.empty((HEIGHT, WIDTH, 3))
time0 = time.perf_counter()
program1(im)
print(time.perf_counter() - time0)

im = (im * im * 255.999).astype(np.uint8)
Image.fromarray(im).save("image.png")
