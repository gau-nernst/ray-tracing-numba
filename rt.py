import numba as nb
import numpy as np


INF = 1e8


@nb.njit(nb.float64[::1](nb.float64[::1]))
def normalize(v):
    return v / np.linalg.norm(v)


@nb.njit(nb.float64[::1](nb.float64[::1], nb.float64[::1]))
def reflect(ray, normal):
    return ray - 2.0 * ray.dot(normal) * normal


@nb.njit
def refract(incident, normal, n1, n2):
    # snell's law
    eta = n1 / n2
    cos1 = -incident.dot(normal)
    eta_cos1 = eta * cos1
    cos2sq = 1 + eta_cos1 * eta_cos1 - eta * eta

    if cos2sq > 1.0 or cos2sq < 0:  # total internal reflection
        return False, incident
    else:
        refracted = eta * incident + (eta_cos1 - np.sqrt(cos2sq)) * normal
        return True, refracted


LIGHT_SOURCE = 0
METAL = 1


@nb.experimental.jitclass
class Sphere:
    center: nb.float64[::1]
    radius: nb.float64
    color: nb.float64[::1]
    material: nb.int64

    def __init__(self, center, radius, color, material):
        self.center = center
        self.radius = radius
        self.color = color
        self.material = material


@nb.njit
def hit_sphere(ray_origin, ray_direction, sphere, t_min, t_max):
    oc = ray_origin - sphere.center
    qb = oc.dot(ray_direction)
    qc = oc.dot(oc) - sphere.radius * sphere.radius
    discriminant = qb * qb - qc

    if discriminant < 0:
        return INF

    sqrt_disc = np.sqrt(discriminant)
    t = -qb - sqrt_disc
    if t_min < t and t < t_max:
        return t

    t = -qb + sqrt_disc
    if t_min < t and t < t_max:
        return t

    return INF


@nb.njit
def hit_many_spheres(ray_origin, ray_direction, spheres):
    idx = 0
    t_max = INF
    hit_something = False
    for i in range(len(spheres)):
        # shadow acne
        t = hit_sphere(ray_origin, ray_direction, spheres[i], 0.0001, t_max)
        if t < t_max:
            t_max = t
            idx = i
            hit_something = True
    return hit_something, t_max, idx
