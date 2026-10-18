import ast
import math


METHODS = ('rrt_star', 'irrt_star', 'rag_bpc_conn_init_rel_only_topk_fallback')


def required_radius(length, width, margin):
    if not all(math.isfinite(v) for v in [length, width, margin]) or min(length, width) <= 0 or margin < 0:
        raise ValueError('invalid robot dimensions/margin')
    return math.hypot(length / 2 + margin, width / 2 + margin)


def planning_radius(config):
    radius = float(config.get('planner_radius_m', -1.0))
    if radius == -1.0:
        return required_radius(config['robot_length_m'], config['robot_width_m'], config['body_margin_m'])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('planner_radius_m must be positive, or -1 for body dimensions')
    return radius


def footprint_covers(footprint, radius, length, width, margin):
    # Require a convex polygon covering all expanded body corners, or a circle.
    parsed = ast.literal_eval(footprint) if footprint.strip() else []
    if not parsed:
        return math.isfinite(radius) and radius + 1e-9 >= required_radius(length, width, margin)
    vertices = [(float(p[0]), float(p[1])) for p in parsed]
    if len(vertices) < 3 or not all(math.isfinite(v) for p in vertices for v in p):
        return False
    def cross(a, b, p):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    turns = [cross(vertices[i], vertices[(i+1)%len(vertices)], vertices[(i+2)%len(vertices)])
             for i in range(len(vertices))]
    if not (all(v >= -1e-9 for v in turns) or all(v <= 1e-9 for v in turns)):
        return False
    area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(vertices,vertices[1:]+vertices[:1]))
    if abs(area) < 1e-9:
        return False
    def inside(p):
        sides = [cross(a, b, p) for a, b in zip(vertices, vertices[1:]+vertices[:1])]
        return all(s >= -1e-9 for s in sides) or all(s <= 1e-9 for s in sides)
    x, y = length/2+margin, width/2+margin
    return all(inside(p) for p in [(x,y),(x,-y),(-x,y),(-x,-y)])


def yaw_of(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def wrap_yaw(value):
    return math.atan2(math.sin(value), math.cos(value))


def resample(points, spacing):
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be finite and positive')
    if len(points) < 2 or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('invalid polyline')
    output = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        dist = math.hypot(b[0]-a[0], b[1]-a[1])
        if dist < 1e-10:
            continue
        steps = max(1, math.ceil(dist/spacing))
        output.extend((a[0]+(b[0]-a[0])*i/steps, a[1]+(b[1]-a[1])*i/steps)
                      for i in range(1, steps))
        output.append(tuple(b))
    if len(output) < 2:
        raise ValueError('zero length path')
    return output


def distance_to_path(point, points):
    best = math.inf
    for a,b in zip(points, points[1:]):
        dx,dy = b[0]-a[0],b[1]-a[1]; denom=dx*dx+dy*dy
        t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denom)) if denom else 0.
        best=min(best,math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy))
    return best
