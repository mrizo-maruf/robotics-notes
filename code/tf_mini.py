"""A minimal re-implementation of the core of ROS 2 tf2, built only from the
book's primitives (SE(3) matrices, unit quaternions, SLERP).

It mirrors tf2's BufferCore (ros2/geometry2, tf2/src/buffer_core.cpp):
  * every frame has exactly one parent; an edge stores T_{parent,child}(t)
  * static edges have no time; dynamic edges keep a time-sorted history
  * lookup(target, source, t) walks source -> root and target -> root and
    returns T_{target,source} = T_{root,target}^{-1} T_{root,source}
  * t = 0 means "latest common time": the minimum over the newest stamps of
    the dynamic edges on the path (static edges do not count)
  * between two stamps: linear interpolation of translation, SLERP of rotation
  * outside the stored interval: ExtrapolationError
Quaternions are stored scalar-first (w, x, y, z) like the rest of the book;
`from_ros` converts a ROS message order (x, y, z, w).
"""
import bisect
import numpy as np
from rigid import q2R, R2q, slerp, T as make_T


class LookupError_(Exception):
    pass


class ExtrapolationError(Exception):
    pass


def from_ros(xyzw):
    x, y, z, w = xyzw
    return np.array([w, x, y, z], float)


class Edge:
    def __init__(self, parent, static):
        self.parent, self.static = parent, static
        self.stamps, self.data = [], []          # data: (translation, quaternion wxyz)

    def add(self, stamp, t, q):
        q = np.asarray(q, float) / np.linalg.norm(q)
        if self.static:
            self.stamps, self.data = [0.0], [(np.asarray(t, float), q)]
            return
        i = bisect.bisect(self.stamps, stamp)
        self.stamps.insert(i, stamp); self.data.insert(i, (np.asarray(t, float), q))

    def latest(self):
        return 0.0 if self.static else self.stamps[-1]

    def at(self, time):
        if self.static:
            t, q = self.data[0]
            return make_T(q2R(q), t)
        if not self.stamps[0] <= time <= self.stamps[-1]:
            raise ExtrapolationError(
                f"lookup at {time:.4f} outside [{self.stamps[0]:.4f}, {self.stamps[-1]:.4f}]")
        i = bisect.bisect_left(self.stamps, time)
        if self.stamps[i] == time:
            t, q = self.data[i]
            return make_T(q2R(q), t)
        (t0, q0), (t1, q1) = self.data[i - 1], self.data[i]
        a = (time - self.stamps[i - 1]) / (self.stamps[i] - self.stamps[i - 1])
        return make_T(q2R(slerp(q0, q1, a)), (1 - a) * t0 + a * t1)   # lerp + slerp, as tf2


class Buffer:
    def __init__(self):
        self.edges = {}                           # child -> Edge

    def set_transform(self, parent, child, stamp, t, q, static=False):
        e = self.edges.get(child)
        if e is None or e.parent != parent:      # one parent per frame (the latest wins, as in tf2)
            e = self.edges[child] = Edge(parent, static)
        e.add(stamp, t, q)

    def chain(self, frame):
        """frames from `frame` up to the root (inclusive)."""
        out = [frame]
        while frame in self.edges:
            frame = self.edges[frame].parent
            out.append(frame)
        return out

    def path(self, target, source):
        cs, ct = self.chain(source), self.chain(target)
        common = next((f for f in cs if f in ct), None)
        if common is None:
            raise LookupError_(f"{source} and {target} are not connected")
        return cs[:cs.index(common)], ct[:ct.index(common)], common

    def latest_common_time(self, target, source):
        up_s, up_t, _ = self.path(target, source)
        stamps = [self.edges[f].latest() for f in up_s + up_t if not self.edges[f].static]
        return min(stamps) if stamps else 0.0

    def lookup(self, target, source, time=0.0):
        """T_{target,source}: maps coordinates in `source` to coordinates in `target`."""
        if time == 0.0:
            time = self.latest_common_time(target, source)
        up_s, up_t, _ = self.path(target, source)
        T_top_source = np.eye(4)
        for f in up_s:                             # accumulate T_{common,source}
            T_top_source = self.edges[f].at(time) @ T_top_source
        T_top_target = np.eye(4)
        for f in up_t:                             # accumulate T_{common,target}
            T_top_target = self.edges[f].at(time) @ T_top_target
        return np.linalg.solve(T_top_target, T_top_source), time
