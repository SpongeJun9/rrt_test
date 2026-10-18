"""Conservative cell-crossing check for an output polyline, outside planner stats.

Integer coordinates denote cell centers. Contact with an occupied cell border
is rejected. Clearance is the same square cell neighborhood as GridMap2D;
this is not a measured real-robot footprint or dynamic-obstacle check.
"""
import math

import numpy as np


def validate_polyline(binary_mask, path, clearance=0):
    mask = np.asarray(binary_mask)
    points = np.asarray(path, dtype=float)
    if mask.ndim != 2 or points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return False, 0, 'invalid shape or fewer than two points'
    if not np.isfinite(points).all() or clearance < 0:
        return False, 0, 'nonfinite coordinates or invalid clearance'
    height, width = mask.shape
    checks = 0

    def cells(value):
        shifted = float(value) + 0.5
        integer = round(shifted)
        if abs(shifted - integer) <= 1e-9:
            return [integer - 1, integer]
        return [math.floor(shifted)]

    def valid(point):
        nonlocal checks
        for x in cells(point[0]):
            for y in cells(point[1]):
                checks += 1
                x0, x1 = x - clearance, x + clearance + 1
                y0, y1 = y - clearance, y + clearance + 1
                if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    return False
                if not np.all(mask[y0:y1, x0:x1] > 0):
                    return False
        return True

    for index, (start, end) in enumerate(zip(points, points[1:])):
        if not valid(start) or not valid(end):
            return False, checks, f'edge {index}: endpoint touches blocked/out-of-map cell'
        delta = end - start
        times = [0.0, 1.0]
        for axis in [0, 1]:
            if abs(delta[axis]) < 1e-12:
                continue
            low, high = sorted([float(start[axis]), float(end[axis])])
            for integer in range(math.ceil(low - 0.5), math.floor(high - 0.5) + 1):
                t = (integer + 0.5 - start[axis]) / delta[axis]
                if 0.0 < t < 1.0:
                    times.append(float(t))
        times = sorted(set(times))
        probes = times + [(a + b) / 2 for a, b in zip(times, times[1:])]
        if not all(valid(start + t * delta) for t in probes):
            return False, checks, f'edge {index}: touches blocked/out-of-map cell'
    return True, checks, 'PASS'
