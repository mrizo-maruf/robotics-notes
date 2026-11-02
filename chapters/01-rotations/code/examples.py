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
