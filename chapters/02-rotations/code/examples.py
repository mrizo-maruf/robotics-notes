import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))  # repo-level shared library
import numpy as np
from scipy.linalg import expm
from rigid import *
np.set_printoptions(precision=4, suppress=True)

print("=== E1 camera mount ===")
R_BC = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
print(R_BC, "det", np.linalg.det(R_BC))
pC = np.array([0.2, -0.1, 3.0])
print("pB", R_BC @ pC, " R^T R", np.allclose(R_BC.T @ R_BC, np.eye(3)))

print("=== E2 relative orientation IMUs ===")
R_W1 = Rz(40 * deg) @ Ry(10 * deg)
R_W2 = R_W1 @ Ry(55 * deg) @ Rx(3 * deg)
R_12 = R_W1.T @ R_W2
print("R_W1\n", R_W1, "\nR_W2\n", R_W2, "\nR_12\n", R_12)
phi = Log(R_12); th = np.linalg.norm(phi)
print("angle deg", th / deg, "axis", phi / th, "tr", np.trace(R_12))
# naive: difference of euler angles
print("euler W1", np.array(zyx_from_R(R_W1)) / deg, "W2", np.array(zyx_from_R(R_W2)) / deg)

print("=== E3 Euler ===")
print("ZYX matrix symbolic check at theta=90:")
for psi, ph in [(30, 50), (0, 20), (-40, -20)]:
    print(psi, ph, "\n", zyx(psi * deg, 90 * deg, ph * deg))
Rt = zyx(10 * deg, 89.5 * deg, 20 * deg)
for d in [np.array([0, 0, 0.3]) * deg, np.array([0.3, 0, 0]) * deg, np.array([0, 0.3, 0]) * deg]:
    Rm = Rt @ Exp(d)
    print("perturb body", d / deg, "->", np.array(zyx_from_R(Rm)) / deg, "geodesic", np.linalg.norm(Log(Rt.T @ Rm)) / deg)
# Euler rates for given body rate at theta=80
def E_inv(phi, th):
    return np.array([[1, np.sin(phi) * np.tan(th), np.cos(phi) * np.tan(th)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi) / np.cos(th), np.cos(phi) / np.cos(th)]])
wb = np.array([0, 0, 1.0])
print("rates (phidot, thdot, psidot) at th=80, phi=0:", E_inv(0, 80 * deg) @ wb)
# verify E (body rates from euler rates) numerically
psi, th, ph = 0.3, 0.7, -0.4
rates = np.array([0.2, -0.5, 0.9])  # phidot, thdot, psidot
h = 1e-6
R0 = zyx(psi, th, ph); R1 = zyx(psi + h * rates[2], th + h * rates[1], ph + h * rates[0])
Wb = R0.T @ (R1 - R0) / h
wb_num = np.array([Wb[2, 1], Wb[0, 2], Wb[1, 0]])
E = np.array([[1, 0, -np.sin(th)], [0, np.cos(ph), np.sin(ph) * np.cos(th)], [0, -np.sin(ph), np.cos(ph) * np.cos(th)]])
print("E check", wb_num, E @ rates, " inv check", E_inv(ph, th) @ (E @ rates), rates)
# yaw average
print("yaw avg naive", (179 + -179) / 2)

print("=== Rodrigues vs expm ===")
w = np.array([0.3, -0.5, 0.8]); print(np.abs(Exp(w) - expm(skew(w))).max())

print("=== E4 rotation error ===")
Rg = Exp(np.array([0.1, -0.2, 0.3])); Re = Exp(np.array([0.12, -0.18, 0.29]))
e = Log(Rg.T @ Re); print("err vec", e, "angle deg", np.linalg.norm(e) / deg,
                          "naive diff norm deg", np.linalg.norm([0.02, 0.02, -0.01]) / deg)
print("Frobenius ||Rg-Re||", np.linalg.norm(Rg - Re), "2*sqrt2*sin(th/2)", 2 * np.sqrt(2) * np.sin(np.linalg.norm(e) / 2))

print("=== E5 rotation averaging ===")
R0 = Rz(90 * deg) @ Rx(-90 * deg)
print("R0\n", R0)
ds = [np.array([2, -1, 0.5]) * deg, np.array([-1, 1.5, 1]) * deg, np.array([0.5, -0.5, -2]) * deg,
      np.array([25, 10, -5]) * deg]
Rs = [R0 @ Exp(d) for d in ds[:3]]
M = sum(Rs) / 3
print("naive mean det", np.linalg.det(M), "orth err", np.linalg.norm(M.T @ M - np.eye(3)))
Rm = Rs[0].copy()
for it in range(6):
    dl = sum(Log(Rm.T @ R) for R in Rs) / 3
    Rm = Rm @ Exp(dl)
    print(" iter", it, "|step| deg", np.linalg.norm(dl) / deg)
print("Karcher mean offset from R0 (deg)", Log(R0.T @ Rm) / deg, "mean of deltas", sum(ds[:3]) / 3 / deg)
U, S, Vt = np.linalg.svd(M); Rc = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
print("chordal", Log(R0.T @ Rc) / deg)
# with an outlier-ish larger
Rs4 = [R0 @ Exp(d) for d in ds]
Rm = Rs4[0].copy()
for it in range(20):
    dl = sum(Log(Rm.T @ R) for R in Rs4) / 4; Rm = Rm @ Exp(dl)
print("4-sample karcher offset deg", Log(R0.T @ Rm) / deg, " mean of deltas", sum(ds) / 4 / deg)

print("=== E6 quaternions ===")
q1 = qaa([0, 0, 1], np.pi / 2); q2 = qaa([1, 0, 0], np.pi / 2)
a = qmul(q1, q2); b = qmul(q2, q1)
print("q1q2", a, "q2q1", b)
for q in (a, b):
    ang = 2 * np.arccos(q[0]); print("  angle", ang / deg, "axis", q[1:] / np.sin(ang / 2))
print("R check", np.allclose(q2R(a), Rz(np.pi / 2) @ Rx(np.pi / 2)), np.allclose(q2R(b), Rx(np.pi / 2) @ Rz(np.pi / 2)))
v = np.array([0, 1, 0.])
p = np.concatenate([[0], v])
print("rotate (0,1,0) by q1q2:", qmul(qmul(a, p), qconj(a)), q2R(a) @ v)
print("R2q roundtrip", R2q(q2R(a)))
# integration of IMU quaternion
w = np.array([0.4, -0.2, 1.0]); dt = 0.005; q = np.array([1, 0, 0, 0.])
for k in range(200):
    q = qmul(q, qaa(w, np.linalg.norm(w) * dt))
print("after 1s q", q, "vs Exp", R2q(Exp(w * 1.0)))

print("=== E7 SLERP yaw wrap ===")
qa = qaa([0, 0, 1], 170 * deg)
qb = R2q(Rz(-170 * deg))
print("qa", qa, "qb", qb, "dot", qa @ qb)
mid_bad = slerp(qa, qb, 0.5, shortest=False); mid_good = slerp(qa, qb, 0.5)
for nm, m in [("bad", mid_bad), ("good", mid_good)]:
    print(nm, m, "yaw deg", np.degrees(np.arctan2(q2R(m)[1, 0], q2R(m)[0, 0])))
# rolling shutter timestamp interpolation example
q0 = R2q(zyx(20 * deg, 5 * deg, -3 * deg)); q1_ = R2q(zyx(32 * deg, 2 * deg, 1 * deg))
qt = slerp(q0, q1_, 0.4)
print("q0", q0, "q1", q1_, "q(0.4)", qt, "euler", np.array(zyx_from_R(q2R(qt))) / deg)
Om = np.arccos(q0 @ q1_); print("Omega deg", Om / deg, "rot angle", 2 * Om / deg)
print("ratio check", np.linalg.norm(Log(q2R(q0).T @ q2R(qt))) / np.linalg.norm(Log(q2R(q0).T @ q2R(q1_))))

print("=== E8 transform chain ===")
T_WB = T(Rz(30 * deg), [2.0, 1.0, 0.0])
R_BC0 = R_BC
R_BCt = Ry(20 * deg) @ R_BC0
T_BC = T(R_BCt, [0.25, 0, 0.6])
print("R_BC tilted\n", R_BCt)
pC = np.array([0.3, 0.2, 2.5, 1])
pB = T_BC @ pC; pW = T_WB @ pB
print("pB", pB, "pW", pW)
T_WC = T_WB @ T_BC; print("T_WC\n", T_WC)
T_CW = Tinv(T_WC); lW = np.array([4.0, 2.5, 0.3, 1]); lC = T_CW @ lW
print("T_CW\n", T_CW, "\nlandmark in C", lC)
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.]])
uv = K @ lC[:3]; print("pixel", uv[:2] / uv[2])

print("=== E9 Kabsch ===")
rng = np.random.default_rng(3)
P = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.15], [0.2, 0.1, 0.05]])
Rtrue = Exp(np.array([0.3, -0.6, 1.1])); ttrue = np.array([0.1, -0.05, 0.8])
Q = (Rtrue @ P.T).T + ttrue + rng.normal(0, 0.002, P.shape)
pc, qc = P.mean(0), Q.mean(0)
H = (P - pc).T @ (Q - qc)
U, S, Vt = np.linalg.svd(H)
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
Rk = Vt.T @ D @ U.T; tk = qc - Rk @ pc
print("Q\n", Q)
print("R err deg", np.linalg.norm(Log(Rtrue.T @ Rk)) / deg, "t err mm", np.linalg.norm(tk - ttrue) * 1000)
print("rmse mm", np.sqrt(np.mean(np.sum((Q - (Rk @ P.T).T - tk) ** 2, 1))) * 1000)
print("Rtrue\n", Rtrue, "\nRk\n", Rk, "tk", tk)

print("=== E10 PoE FK ===")
L0, L1, L2 = 0.3, 0.4, 0.3
def S(w, q):
    w = np.array(w, float); q = np.array(q, float)
    return np.concatenate([-np.cross(w, q), w])
S1 = S([0, 0, 1], [0, 0, 0]); S2 = S([0, -1, 0], [0, 0, L0]); S3 = S([0, -1, 0], [L1, 0, L0])
print("S1", S1, "S2", S2, "S3", S3)
M = T(np.eye(3), [L1 + L2, 0, L0])
th = np.array([30, 45, -60]) * deg
Tfk = expm(hat6(S1) * th[0]) @ expm(hat6(S2) * th[1]) @ expm(hat6(S3) * th[2]) @ M
print("Tfk\n", Tfk)
r = L1 * np.cos(th[1]) + L2 * np.cos(th[1] + th[2]); z = L0 + L1 * np.sin(th[1]) + L2 * np.sin(th[1] + th[2])
print("geom", r * np.cos(th[0]), r * np.sin(th[0]), z)
print("ExpSE3 vs expm", np.abs(ExpSE3(S2 * th[1]) - expm(hat6(S2) * th[1])).max())
# space Jacobian at th
J1 = S1
J2 = Ad(expm(hat6(S1) * th[0])) @ S2
J3 = Ad(expm(hat6(S1) * th[0]) @ expm(hat6(S2) * th[1])) @ S3
Js = np.column_stack([J1, J2, J3]); print("Js\n", Js)
# check numerically: tip velocity = v_s + w_s x p
dth = np.array([0.1, -0.2, 0.3])
Vs = Js @ dth; p = Tfk[:3, 3]
vtip = Vs[:3] + np.cross(Vs[3:], p)
h = 1e-6
th2 = th + h * dth
Tfk2 = expm(hat6(S1) * th2[0]) @ expm(hat6(S2) * th2[1]) @ expm(hat6(S3) * th2[2]) @ M
print("tip vel", vtip, "numeric", (Tfk2[:3, 3] - p) / h)

print("=== E11 Adjoint camera velocity ===")
Vb = np.array([1.0, 0, 0, 0, 0, 0.5])
T_BCm = T(R_BC0, [0.3, 0, 0.5])
Vc = Ad(Tinv(T_BCm)) @ Vb
print("Vc (camera body twist, camera coords)", Vc)
print("point vel in B", Vb[:3] + np.cross(Vb[3:], [0.3, 0, 0.5]), "-> in C", R_BC0.T @ (Vb[:3] + np.cross(Vb[3:], [0.3, 0, 0.5])))

print("=== E12 screw from pose ===")
T2 = T(Rz(np.pi / 2), [1, 1, 0.5])
xi = LogSE3(T2); print("log", xi)
th_ = np.linalg.norm(xi[3:]); w_ = xi[3:] / th_; v_ = xi[:3] / th_
print("theta", th_, "w", w_, "v", v_, "pitch", w_ @ v_, "q=w x v", np.cross(w_, v_))
print("check Exp", np.abs(ExpSE3(xi) - T2).max())
mid = ExpSE3(0.5 * xi); print("screw midpoint\n", mid)


# =====================================================================================
#  Section "Frames in practice: tf2 in ROS 2"   (all numbers in that section)
# =====================================================================================
from tf_mini import Buffer, ExtrapolationError, from_ros
import rigid as rg
print("\n\n######## tf2 section ########")


def setRPY(r, p, y):  # copied from tf2/LinearMath/Quaternion.hpp, returns (x,y,z,w)
    hy, hp, hr = y / 2, p / 2, r / 2
    cy, sy, cp, sp, cr, sr = np.cos(hy), np.sin(hy), np.cos(hp), np.sin(hp), np.cos(hr), np.sin(hr)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


errs = [np.abs(q2R(from_ros(setRPY(*a))) - zyx(a[2], a[1], a[0])).max() for a in np.random.default_rng(1).uniform(-3, 3, (500, 3))]
print("tf2 setRPY(r,p,y) == Rz(y)Ry(p)Rx(r):", max(errs))
R_opt = q2R(from_ros(setRPY(-np.pi / 2, 0, -np.pi / 2)))
print("optical rotation from rpy=(-pi/2, 0, -pi/2):\n", np.round(R_opt, 6), "\n == R_BC of ch.2 example:", np.allclose(R_opt, R_BC))

# ---- illustrative FreiCAR/CARLA tree (structure from the user's view_frames; numbers illustrative)
v, w, z0 = 1.5, 0.6, 0.0
t_first, t_last, dt = 2058.548925, 2068.498926, 0.05
stamps = t_first + dt * np.arange(int(round((t_last - t_first) / dt)) + 1)
t_last = stamps[-1]
def veh(t):  # map -> freicar_1 : planar arc (exact SE(2) exponential), start (5,3,20deg)
    T0 = rg.T2(np.radians(20), [5.0, 3.0]); M = T0 @ rg.ExpSE2([v * (t - t_first), 0, w * (t - t_first)])
    return T(Rz(np.arctan2(M[1, 0], M[0, 0])), [M[0, 2], M[1, 2], z0])
T_veh_base = T(np.eye(3), [-0.15, 0.0, -0.06])            # freicar_1 -> base_link (static)
T_base_lmount = T(np.eye(3), [0.20, 0.0, 0.14])            # URDF: base_link -> laser_mount_link
T_veh_lidar_true = T(Rz(np.radians(1.0)), [0.07, 0.0, 0.10])  # where CARLA actually spawned the lidar
T_veh_depth = T(np.eye(3), [0.25, 0.0, 0.20])
buf = Buffer()
buf.set_transform("freicar_1", "freicar_1/base_link", 0, T_veh_base[:3, 3], R2q(T_veh_base[:3, :3]), static=True)
buf.set_transform("freicar_1/base_link", "freicar_1/laser_mount_link", 0, T_base_lmount[:3, 3], R2q(np.eye(3)), static=True)
for s in stamps:
    Mv = veh(s); buf.set_transform("map", "freicar_1", s, Mv[:3, 3], R2q(Mv[:3, :3]))
    Ml = Mv @ T_veh_lidar_true; buf.set_transform("map", "freicar_1/lidar", s, Ml[:3, 3], R2q(Ml[:3, :3]))
    if s < stamps[-1] - 1e-9:   # depth camera edge lags one sample (50 ms), as in the recorded tree
        Md = Mv @ T_veh_depth; buf.set_transform("map", "freicar_1/camera_depth_front", s, Md[:3, 3], R2q(Md[:3, :3]))
print("chain lidar:", buf.chain("freicar_1/lidar"), " chain base:", buf.chain("freicar_1/base_link"))
M, tl = buf.lookup("freicar_1/base_link", "freicar_1/lidar")
print(f"lookup(base_link, lidar) @ latest common {tl:.6f}: t = {M[:3,3]}, yaw = {np.degrees(np.arctan2(M[1,0],M[0,0])):.3f} deg")
expect = np.linalg.inv(T_veh_base) @ T_veh_lidar_true
print("   expected (static truth):", expect[:3, 3], " match:", np.allclose(M, expect))
D, _ = buf.lookup("freicar_1/laser_mount_link", "freicar_1/lidar")
print(f"URDF vs CARLA discrepancy T_(laser_mount, lidar): t = {D[:3,3]}, yaw = {np.degrees(np.arctan2(D[1,0],D[0,0])):.3f} deg")
pt = np.array([10.0, 0, 0, 1]); print("point 10 m ahead of lidar, lateral offset caused by the 1 deg error:", (D @ pt)[1])
print("latest common time base_link<->depth:", f"{buf.latest_common_time('freicar_1/base_link', 'freicar_1/camera_depth_front'):.6f}",
      " base_link<->lidar:", f"{buf.latest_common_time('freicar_1/base_link', 'freicar_1/lidar'):.6f}")
try:
    buf.lookup("freicar_1/base_link", "freicar_1/camera_depth_front", t_last)
except ExtrapolationError as e:
    print("lookup at newest vehicle stamp ->", e)
# mixing stamps by hand: vehicle pose at t_last, camera pose at t_last-0.05
Tb_new = buf.edges["freicar_1"].at(t_last) @ T_veh_base
Tc_old = buf.edges["freicar_1/camera_depth_front"].at(stamps[-2])
Mix = np.linalg.solve(Tb_new, Tc_old); Good = np.linalg.inv(T_veh_base) @ T_veh_depth
dE = np.linalg.solve(Good, Mix)
print(f"stamp-mixing error in T_(base, depth): |dt| = {np.linalg.norm(dE[:3,3])*100:.2f} cm, rot = {np.degrees(np.linalg.norm(Log(dE[:3,:3]))):.3f} deg;"
      f" lateral error of a point 10 m ahead = {abs((dE @ pt)[1])*100:.1f} cm")
# interpolation: lerp+slerp vs exact arc
for gap, name in [(0.05, "20 Hz"), (1.0, "1 Hz")]:
    ta, tb = 2060.0, 2060.0 + gap; tm = (ta + tb) / 2
    Ma, Mb, Mtrue = veh(ta), veh(tb), veh(tm)
    lin = (Ma[:3, 3] + Mb[:3, 3]) / 2
    print(f"{name}: midpoint error of lerp+slerp vs true arc = {np.linalg.norm(lin - Mtrue[:3,3])*1000:.3f} mm;"
          f" sagitta formula r(1-cos(w*gap/2)) = {v/w*(1-np.cos(w*gap/2))*1000:.3f} mm")
wall = 1793804439.4143991
print(f"use_sim_time=false: now - newest stamp = {wall - t_last:.0f} s = {(wall - t_last)/3.15576e7:.1f} years")

# ---- map / odom / base_link (planar, REP 105)
T_odom_base = rg.T2(np.radians(30), [10.0, 2.0]); T_map_base = rg.T2(np.radians(33), [10.4, 1.7])
T_map_odom = T_map_base @ rg.T2inv(T_odom_base)
print("\nT_map_odom = T_map_base T_odom_base^-1 :", T_map_odom[0, 2], T_map_odom[1, 2], np.degrees(np.arctan2(T_map_odom[1, 0], T_map_odom[0, 0])))
T_odom_base2 = T_odom_base @ rg.T2(0, [1.0, 0]); T_map_base2 = T_map_odom @ T_odom_base2
print("after 1 m of odometry: base in odom", T_odom_base2[:2, 2], " base in map", T_map_base2[:2, 2], np.degrees(np.arctan2(T_map_base2[1,0],T_map_base2[0,0])))

# ---- quaternion discontinuity vs 6D (in-practice subsection of the quaternion section)
q1, q2 = R2q(Rz(np.radians(179.5))), R2q(Rz(np.radians(-179.5)))
print("\nRz(179.5) vs Rz(-179.5): angle between =", np.degrees(np.linalg.norm(Log(Rz(np.radians(179.5)).T @ Rz(np.radians(-179.5))))), "deg")
print("  quaternions (w>=0):", q1, q2, " |q1-q2| =", np.linalg.norm(q1 - q2), " 1-|q1.q2| =", 1 - abs(q1 @ q2))
s6 = lambda R: np.r_[R[:, 0], R[:, 1]]
print("  6D (first two columns) distance:", np.linalg.norm(s6(Rz(np.radians(179.5))) - s6(Rz(np.radians(-179.5)))))

# ---- in-practice checks: scipy Rotation == Exp/Log; scipy quaternion order; URDF-style FK == PoE
from scipy.spatial.transform import Rotation as Rot
import scipy
phi = np.array([0.3, -0.5, 0.8])
print("\nscipy", scipy.__version__, ": from_rotvec == Exp:", np.abs(Rot.from_rotvec(phi).as_matrix() - Exp(phi)).max(),
      "; as_rotvec == Log:", np.abs(Rot.from_matrix(Exp(phi)).as_rotvec() - phi).max())
print("scipy as_quat() default (scalar last):", Rot.from_rotvec([0, 0, np.pi / 2]).as_quat(),
      " scalar_first=True:", Rot.from_rotvec([0, 0, np.pi / 2]).as_quat(scalar_first=True))
# URDF-style: T_parent_child(q) = T_origin * Exp(q * axis) for the 3-DOF elbow arm of the PoE example
L0, L1, L2 = 0.3, 0.4, 0.3
def urdf_joint(xyz, axis, q):
    return T(np.eye(3), xyz) @ T(Exp(q * np.asarray(axis, float)), [0, 0, 0])
qs = np.radians([30, 45, -60])
Tu = urdf_joint([0, 0, 0], [0, 0, 1], qs[0]) @ urdf_joint([0, 0, L0], [0, -1, 0], qs[1]) @ \
     urdf_joint([L1, 0, 0], [0, -1, 0], qs[2]) @ T(np.eye(3), [L2, 0, 0])
print("URDF-style chain tool position:", Tu[:3, 3], " equals PoE result:", np.allclose(Tu, Tfk))
