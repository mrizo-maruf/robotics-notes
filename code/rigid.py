import numpy as np

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Exp(phi):
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K

def Log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        B = (R + np.eye(3)) / 2
        i = np.argmax(np.diag(B))
        w = B[:, i] / np.sqrt(B[i, i])
        return th * w
    W = (R - R.T) / (2 * np.sin(th))
    return th * np.array([W[2, 1], W[0, 2], W[1, 0]])

def zyx(psi, th, phi):
    return Rz(psi) @ Ry(th) @ Rx(phi)

def zyx_from_R(R):
    th = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    psi = np.arctan2(R[1, 0], R[0, 0])
    phi = np.arctan2(R[2, 1], R[2, 2])
    return psi, th, phi

# quaternions, Hamilton, scalar first
def qmul(a, b):
    w1, v1 = a[0], a[1:]
    w2, v2 = b[0], b[1:]
    return np.concatenate([[w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qaa(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])

def q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def R2q(R):
    # Shepperd
    tr = np.trace(R)
    d = np.array([tr, R[0, 0], R[1, 1], R[2, 2]])
    i = np.argmax(d)
    if i == 0:
        w = np.sqrt(1 + tr) / 2
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    elif i == 1:
        x = np.sqrt(1 + 2 * R[0, 0] - tr) / 2
        q = np.array([(R[2, 1] - R[1, 2]) / (4 * x), x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)])
    elif i == 2:
        y = np.sqrt(1 + 2 * R[1, 1] - tr) / 2
        q = np.array([(R[0, 2] - R[2, 0]) / (4 * y), (R[0, 1] + R[1, 0]) / (4 * y), y, (R[1, 2] + R[2, 1]) / (4 * y)])
    else:
        z = np.sqrt(1 + 2 * R[2, 2] - tr) / 2
        q = np.array([(R[1, 0] - R[0, 1]) / (4 * z), (R[0, 2] + R[2, 0]) / (4 * z), (R[1, 2] + R[2, 1]) / (4 * z), z])
    return q if q[0] >= 0 else -q

def slerp(q0, q1, t, shortest=True):
    d = q0 @ q1
    if shortest and d < 0:
        q1, d = -q1, -d
    d = np.clip(d, -1, 1)
    Om = np.arccos(d)
    if Om < 1e-9:
        return q0
    return (np.sin((1 - t) * Om) * q0 + np.sin(t * Om) * q1) / np.sin(Om)

def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def Tinv(M):
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)

def V_mat(phi):
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-12:
        return np.eye(3) + K / 2
    return np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K

def ExpSE3(xi):  # xi = (rho, phi) i.e. (v*theta, w*theta)
    rho, phi = xi[:3], xi[3:]
    return T(Exp(phi), V_mat(phi) @ rho)

def LogSE3(M):
    phi = Log(M[:3, :3])
    rho = np.linalg.solve(V_mat(phi), M[:3, 3])
    return np.concatenate([rho, phi])

def hat6(xi):
    M = np.zeros((4, 4))
    M[:3, :3] = skew(xi[3:])
    M[:3, 3] = xi[:3]
    return M

def Ad(M):
    R, t = M[:3, :3], M[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(t) @ R
    A[3:, 3:] = R
    return A

deg = np.pi / 180


# ===================================================================== SE(2)
# Planar twists are ordered linear-first, S = (v_x, v_y, omega), as in 3-D.
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])  # planar rotation generator, J @ J = -I


def R2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def T2(th, t):
    M = np.eye(3)
    M[:2, :2] = R2(th)
    M[:2, 2] = t
    return M


def hat2(S):
    """(v_x, v_y, w) -> 3x3 element of se(2)."""
    vx, vy, w = S
    return np.array([[0.0, -w, vx], [w, 0.0, vy], [0.0, 0.0, 0.0]])


def vee2(M):
    """'vexa': 3x3 element of se(2) -> (v_x, v_y, w)."""
    return np.array([M[0, 2], M[1, 2], M[1, 0]])


def V2(th):
    """Translation coupling of the SE(2) exponential: (sin th/th) I + ((1-cos th)/th) J."""
    if abs(th) < 1e-6:
        a, b = 1 - th**2 / 6, th / 2 - th**3 / 24
    else:
        a, b = np.sin(th) / th, (1 - np.cos(th)) / th
    return a * np.eye(2) + b * J2


def V2inv(th):
    """(th/2) cot(th/2) I - (th/2) J ; valid for |th| < 2 pi."""
    if abs(th) < 1e-6:
        c = 1 - th**2 / 12
    else:
        c = (th / 2) / np.tan(th / 2)
    return c * np.eye(2) - (th / 2) * J2


def ExpSE2(S):
    """Closed-form exp([S]) for S = (v_x, v_y, w)."""
    v, w = np.asarray(S[:2], float), float(S[2])
    return T2(w, V2(w) @ v)


def LogSE2(M):
    """Principal logarithm: heading in (-pi, pi]."""
    th = np.arctan2(M[1, 0], M[0, 0])
    v = V2inv(th) @ M[:2, 2]
    return np.array([v[0], v[1], th])


def T2inv(M):
    R, t = M[:2, :2], M[:2, 2]
    out = np.eye(3)
    out[:2, :2] = R.T
    out[:2, 2] = -R.T @ t
    return out
