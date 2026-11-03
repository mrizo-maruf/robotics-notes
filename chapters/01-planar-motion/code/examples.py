"""Every number quoted in chapter 1's worked examples is computed here.

Run:  make check CH=01-planar-motion
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))  # repo-level shared library
import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
from rigid import *

np.set_printoptions(precision=5, suppress=True)
Tx = lambda d: T2(0.0, [d, 0.0])
TR = lambda a: T2(a, [0.0, 0.0])


def G_turn(r):
    return hat2([r, 0.0, 1.0])


# ------------------------------------------------------------------ 5.1 one step
print("=== one step: move-then-turn vs exact arc (r = 1) ===")
for dth in [0.5, 0.1, 0.01]:
    step = Tx(dth) @ TR(dth)
    arc = ExpSE2([dth, 0, dth])
    err = np.linalg.norm(step[:2, 2] - arc[:2, 2])
    print(f"dth={dth:5}: step t={step[:2,2]}, arc t={arc[:2,2]}, |err|={err:.3e}, r dth^2/2={dth**2/2:.3e}")

# ------------------------------------------------------------------ 5.3 convergence
print("\n=== (Delta T)^n -> exp(theta G), r = 1, theta = pi/2 ===")
th, r = np.pi / 2, 1.0
Ttrue = expm(th * G_turn(r))
print("exp(theta G) =\n", Ttrue)
rows = []
for n in [1, 2, 4, 8, 16, 64, 256, 1024]:
    d = th / n
    first = np.linalg.matrix_power(np.eye(3) + d * G_turn(r), n)
    mt = np.linalg.matrix_power(Tx(r * d) @ TR(d), n)
    sym = np.linalg.matrix_power(TR(d / 2) @ Tx(r * d) @ TR(d / 2), n)
    e = [np.linalg.norm(M[:2, 2] - Ttrue[:2, 2]) for M in (first, mt, sym)]
    rows.append((n, *e, np.linalg.det(first[:2, :2])))
    print(f"n={n:5d}: |err| I+dG={e[0]:.3e}  move-turn={e[1]:.3e}  half-move-half={e[2]:.3e}  det(rot block of (I+dG)^n)={np.linalg.det(first[:2,:2]):.6f}")
n1, n2 = 64, 1024
print("slopes (log err / log n) between n=64 and 1024:",
      [np.log(rows[7][k] / rows[5][k]) / np.log(n2 / n1) for k in (1, 2, 3)])

# ------------------------------------------------------------------ 5.4 planar rotation
print("\n=== e^{theta J} ===")
print("J^2 =", J2 @ J2, " max|expm(th J) - R2(th)| =", max(np.abs(expm(a * J2) - R2(a)).max() for a in np.linspace(-7, 7, 41)))

# ------------------------------------------------------------------ 5.5 closed form + centre
print("\n=== slide check r=1, theta=pi/2 ===")
M = ExpSE2([np.pi / 2, 0, np.pi / 2]); print(M, "heading deg", np.degrees(np.arctan2(M[1, 0], M[0, 0])))
C = np.array([0.0, 1.0])
TC = T2(0, C); conj = TC @ TR(np.pi / 2) @ T2inv(TC)
print("T_C T_R T_C^-1 == exp:", np.allclose(conj, M))
for S in [(0.4, 0.0, 0.5), (0.3, 0.2, 0.5)]:
    v, w = np.array(S[:2]), S[2]
    Cc = J2 @ v / w
    print(f"twist {S}: centre C = J v / w = {Cc}, check v + w J C = {v + w * J2 @ Cc}, radius |C| = {np.linalg.norm(Cc):.3f}")
# t(theta) vs theta*v
for th_ in [np.pi / 2, np.pi]:
    print(f"theta={th_:.4f}: t = {ExpSE2([th_*1, 0, th_])[:2,2]}, theta*v = {[th_, 0]}")
print("V2(pi/2) =\n", V2(np.pi / 2))

# ------------------------------------------------------------------ 5.6 diff drive odometry
print("\n=== differential-drive odometry, exact exp vs Euler vs RK2 ===")
rw, b, f = 0.035, 0.30, 10.0           # wheel radius [m], track width [m], encoder rate [Hz]
dt, Tend = 1 / f, 60.0
N = int(Tend * f)
tk = np.arange(N) * dt
v_cmd = 0.6 + 0.0 * tk                  # m/s
w_cmd = 0.8 * np.cos(2 * np.pi * (tk + dt / 2) / 8.0)   # rad/s, slalom
wl = (v_cmd - w_cmd * b / 2) / rw       # wheel angular speeds [rad/s], held constant over each sample
wr = (v_cmd + w_cmd * b / 2) / rw
dsl, dsr = wl * rw * dt, wr * rw * dt  # wheel arc lengths per sample
ds, dth_ = (dsr + dsl) / 2, (dsr - dsl) / b
print(f"max |dtheta| per sample = {np.abs(dth_).max():.3f} rad, ds = {ds[0]:.3f} m")


def truth_rhs(t, x):
    k = min(int(t / dt), N - 1)
    return [v_cmd[k] * np.cos(x[2]), v_cmd[k] * np.sin(x[2]), w_cmd[k]]


xs = [np.zeros(3)]
for k in range(N):  # integrate each zero-order-hold interval separately with tight tolerance
    sol = solve_ivp(truth_rhs, (k * dt, (k + 1) * dt), xs[-1], rtol=1e-11, atol=1e-12, max_step=dt / 4)
    xs.append(sol.y[:, -1])
xs = np.array(xs)
Te, xe, xr = np.eye(3), [np.zeros(3)], [np.zeros(3)]
pe = np.zeros(3); pr = np.zeros(3); TT = [np.eye(3)]
for k in range(N):
    Te = Te @ ExpSE2([ds[k], 0, dth_[k]]); TT.append(Te)
    pe = pe + np.array([ds[k] * np.cos(pe[2]), ds[k] * np.sin(pe[2]), dth_[k]]); xe.append(pe)
    pr = pr + np.array([ds[k] * np.cos(pr[2] + dth_[k] / 2), ds[k] * np.sin(pr[2] + dth_[k] / 2), dth_[k]]); xr.append(pr)
xexp = np.array([[M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0])] for M in TT])
xe, xr = np.array(xe), np.array(xr)
for name, X in [("exp", xexp), ("Euler", xe), ("RK2", xr)]:
    err = np.linalg.norm(X[:, :2] - xs[:, :2], axis=1)
    print(f"{name:6s}: final pos err {err[-1]*100:.4f} cm, max {err.max():.2e} m")
L = ds.sum(); print(f"path length {L:.1f} m; final truth pose {xs[-1]}")
# rate sweep (same commands, different encoder rate): only the integrators change
for fr in [50.0, 10.0, 2.0]:
    h = 1 / fr; n_ = int(Tend * fr); t_ = np.arange(n_) * h
    w_ = 0.8 * np.cos(2 * np.pi * (t_ + h / 2) / 8.0); v_ = 0.6 + 0 * t_
    Tq = np.eye(3); pe_ = np.zeros(3); pr_ = np.zeros(3); me = mr = 0.0
    for k in range(n_):
        d, a = v_[k] * h, w_[k] * h
        Tq = Tq @ ExpSE2([d, 0, a])
        pe_ = pe_ + [d * np.cos(pe_[2]), d * np.sin(pe_[2]), a]
        pr_ = pr_ + [d * np.cos(pr_[2] + a / 2), d * np.sin(pr_[2] + a / 2), a]
        me = max(me, np.linalg.norm(pe_[:2] - Tq[:2, 2])); mr = max(mr, np.linalg.norm(pr_[:2] - Tq[:2, 2]))
    print(f"rate {fr:4.0f} Hz: max |Euler - exact| = {me*100:.3f} cm, max |RK2 - exact| = {mr*100:.4f} cm")
np.savez(os.path.join(HERE, "odom.npz"), truth=xs, exp=xexp, euler=xe, rk2=xr, t=np.r_[tk, Tend])

# ------------------------------------------------------------------ 5.7 logarithms
print("\n=== logarithm: 3/4 left turn r=1 ===")
M = ExpSE2([1.5 * np.pi, 0, 1.5 * np.pi])
print("T =\n", M)
S = LogSE2(M); print("principal log vexa =", S, " scipy logm:", vee2(logm(M).real))
v, w = S[:2], S[2]
print("centre of log twist J v / w =", J2 @ v / w, "; unit-rate twist", S / abs(w), "; arc length", abs(w) * np.linalg.norm(J2 @ v / w))
print("full loop exp(2 pi G):\n", ExpSE2([2 * np.pi, 0, 2 * np.pi]), "log ->", LogSE2(ExpSE2([2 * np.pi, 0, 2 * np.pi])))
print("V2inv check:", np.allclose(V2inv(1.234) @ V2(1.234), np.eye(2)))
# pose-graph style relative-pose error
Ta = T2(np.radians(30), [2.0, 1.0]); Tb = T2(np.radians(41), [2.9, 1.6])
Zab = T2(np.radians(10), [1.05, 0.05])
e = LogSE2(T2inv(Zab) @ T2inv(Ta) @ Tb)
print("pose-graph residual Log(Z^-1 Ta^-1 Tb) =", e, " (m, m, rad); heading deg", np.degrees(e[2]))
print("Ta^-1 Tb =\n", T2inv(Ta) @ Tb)

# ------------------------------------------------------------------ 5.8 bracket / parallel parking
print("\n=== Lie bracket: forward/turn manoeuvre ===")
A, B = hat2([1, 0, 0]), hat2([0, 0, 1])
br = A @ B - B @ A
print("[A,B] = vexa ->", vee2(br))
for eps in [0.3, 0.2, 0.1]:
    Mnet = expm(eps * A) @ expm(eps * B) @ expm(-eps * A) @ expm(-eps * B)
    print(f"eps={eps}: net (x,y,th) = {Mnet[0,2]:.5f}, {Mnet[1,2]:.5f}, {np.arctan2(Mnet[1,0],Mnet[0,0]):.2e};  eps^2*[A,B] -> {eps**2*vee2(br)}")
# repeated manoeuvre: K cycles
eps, K = 0.2, 10
Mk = np.linalg.matrix_power(expm(eps * A) @ expm(eps * B) @ expm(-eps * A) @ expm(-eps * B), K)
print(f"{K} cycles eps={eps}: net pose {Mk[0,2]:.4f}, {Mk[1,2]:.4f}, heading {np.degrees(np.arctan2(Mk[1,0],Mk[0,0])):.3f} deg; distance driven {4*eps*K/2:.1f} m forward+back, rotation {2*eps*K:.1f} rad")
print("bracket of two unit rotations about C1=(0,0), C2=(1,0):",
      vee2(hat2([0, 0, 1]) @ hat2(np.r_[-J2 @ [1, 0], 1]) - hat2(np.r_[-J2 @ [1, 0], 1]) @ hat2([0, 0, 1])))
