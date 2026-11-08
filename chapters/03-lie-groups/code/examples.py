"""Every number quoted in chapter 3 (Lie groups) is computed here.
Run:  make check CH=03-lie-groups
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))  # repo-level shared library
import numpy as np
from rigid import *

np.set_printoptions(precision=4, suppress=True)
CHI2_3_95 = 7.814727903251178
yawdeg = lambda M: np.degrees(np.arctan2(M[1, 0], M[0, 0]))

# ------------------------------------------------------------------ 3.1 groups
print("=== groups ===")
z1, z2 = np.exp(1j * 0.7), np.exp(1j * 1.1)
print("S1: z1 z2 angle =", np.angle(z1 * z2), " R(0.7)R(1.1) == R(1.8):", np.allclose(R2(0.7) @ R2(1.1), R2(1.8)))
print("SO(3) not commutative: |RxRz - RzRx| =", np.linalg.norm(Rx(np.pi/2) @ Rz(np.pi/2) - Rz(np.pi/2) @ Rx(np.pi/2)))

# ------------------------------------------------------------------ 3.2 tangent space
print("\n=== velocities: local vs global ===")
R0 = Exp(np.array([0.2, -0.4, 1.0])); wb = np.array([0.1, 0.0, 0.5]); h = 1e-6
Rdot = (R0 @ Exp(wb * h) - R0) / h
loc = R0.T @ Rdot; glo = Rdot @ R0.T
print("R^T Rdot skew? ", np.allclose(loc, -loc.T, atol=1e-5), " vee =", [loc[2, 1], loc[0, 2], loc[1, 0]])
print("Rdot R^T vee =", [glo[2, 1], glo[0, 2], glo[1, 0]], " R0 wb =", R0 @ wb)
q = R2q(R0); qd = (R2q(R0 @ Exp(wb * h)) - q) / h
print("quaternion: q . qdot =", q @ qd, "(tangent to S^3: orthogonal to q)")

# ------------------------------------------------------------------ 3.3 exp, plus, minus
print("\n=== plus / minus on SE(2) ===")
X = T2(np.radians(30), [2.0, 1.0]); tau = np.array([1.0, 0.0, 0.5])
Xr, Xl = rplus("SE2", X, tau), lplus("SE2", X, tau)
print("X (+) tau  (right):", Xr[:2, 2], f"{yawdeg(Xr):.2f} deg")
print("tau (+) X  (left): ", Xl[:2, 2], f"{yawdeg(Xl):.2f} deg")
print("Ad_X tau =", Ad_SE2(X) @ tau, " Exp(Ad_X tau) X == X Exp(tau):", np.allclose(lplus("SE2", X, Ad_SE2(X) @ tau), Xr))
print("(X (+) tau) (-) X =", rminus("SE2", Xr, X))
print("Exp(t tau) == Exp(tau)^t for t=3:", np.allclose(ExpSE2(3 * tau), np.linalg.matrix_power(ExpSE2(tau), 3)))
print("Ad_X =\n", Ad_SE2(X))
xi_b = np.array([0.6, 0.0, 0.4]); print("body twist", xi_b, "-> spatial (world) twist Ad_X xi_b =", Ad_SE2(X) @ xi_b)
v_origin_world = X[:2, :2] @ xi_b[:2]
print("  velocity of robot origin in world = R v_b =", v_origin_world, "; spatial v = v_origin - w J t:", v_origin_world - xi_b[2] * J2 @ X[:2, 2])

# ------------------------------------------------------------------ 3.5 Jacobians
print("\n=== Jacobians ===")
phi = np.array([0.3, -0.5, 0.8])
print("Jr_SO3(phi) =\n", Jr_SO3(phi))
print("FD check Jr:", np.abs(num_rjac(lambda t: Exp(t), phi, "R", "SO3") - Jr_SO3(phi)).max())
print("small-angle Jr ~ I - [phi]/2:", np.abs(Jr_SO3(1e-3 * phi) - (np.eye(3) - 0.5 * skew(1e-3 * phi))).max())
print("Jl_SO3 == V of SE(3) exp:", np.abs(Jl_SO3(phi) - V_mat(phi)).max())
rng = np.random.default_rng(7); W = {}
def upd(k, v): W[k] = max(W.get(k, 0), v)
for _ in range(200):
    X2, Y2, t2 = ExpSE2(rng.normal(size=3)), ExpSE2(rng.normal(size=3)), rng.normal(size=3)
    R = Exp(rng.normal(size=3)); p = rng.normal(size=3)
    upd("inverse", np.abs(num_rjac(T2inv, X2, "SE2", "SE2") + Ad_SE2(X2)).max())
    upd("compose_X", np.abs(num_rjac(lambda A: A @ Y2, X2, "SE2", "SE2") - Ad_SE2(T2inv(Y2))).max())
    upd("compose_Y", np.abs(num_rjac(lambda B: X2 @ B, Y2, "SE2", "SE2") - np.eye(3)).max())
    upd("Exp SE2", np.abs(num_rjac(ExpSE2, t2, "R", "SE2") - Jr_SE2(t2)).max())
    upd("rplus_X", np.abs(num_rjac(lambda A: rplus("SE2", A, t2), X2, "SE2", "SE2") - Ad_SE2(ExpSE2(-t2))).max())
    upd("act_SO3", np.abs(num_rjac(lambda Q: Q @ p, R, "SO3") + R @ skew(p)).max())
    upd("invact_SO3", np.abs(num_rjac(lambda Q: Q.T @ p, R, "SO3") - skew(R.T @ p)).max())
print("max |FD - closed form| over 200 random draws:", {k: f"{v:.1e}" for k, v in W.items()})

# Wahba / attitude from vector observations by Gauss-Newton on SO(3)
print("\n=== attitude from vector observations (Gauss-Newton on SO(3)) ===")
a = np.array([[0, 0, -1.0], [0.48, 0.0, -0.88], [0.6, 0.8, 0.0], [-0.3, 0.5, 0.81]])
a = a / np.linalg.norm(a, axis=1, keepdims=True)
Rtrue = Exp(np.array([0.4, -0.9, 2.2]))
rng = np.random.default_rng(11)
y = (Rtrue.T @ a.T).T + rng.normal(0, 0.01, a.shape)        # body-frame measurements
Rk = np.eye(3); hist = []
for it in range(8):
    r = np.concatenate([y[i] - Rk.T @ a[i] for i in range(len(a))])
    Jm = np.vstack([skew(Rk.T @ a[i]) for i in range(len(a))])   # d(R^T a)/dR (right perturbation)
    d = np.linalg.solve(Jm.T @ Jm, Jm.T @ r)
    Rk = Rk @ Exp(d)
    hist.append((it + 1, np.degrees(np.linalg.norm(Log(Rtrue.T @ Rk))), np.linalg.norm(d), r @ r))
for hrow in hist[:6]:
    print(f" iter {hrow[0]}: error to truth {hrow[1]:.4f} deg, |step| {hrow[2]:.2e}, cost before step {hrow[3]:.3e}")
H = (a).T @ y  # SVD (Kabsch/Wahba) solution of min sum |y_i - R^T a_i|^2, i.e. R^T maps a->y
U, S, Vt = np.linalg.svd(a.T @ y); Rt_svd = (Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T)
print(" GN vs SVD closed form: ", np.degrees(np.linalg.norm(Log(Rk.T @ Rt_svd.T))), "deg")

# ------------------------------------------------------------------ 3.6 banana
print("\n=== banana distribution ===")
def banana(N=5000, steps=20, d=0.1, sv=0.005, sl=1e-4, sw=0.1, seed=3):
    rng = np.random.default_rng(seed)
    Q = np.diag([sv**2, sl**2, sw**2]); u = np.array([d, 0, 0.0])
    Xs = np.repeat(np.eye(3)[None], N, axis=0); Xbar = np.eye(3); P = np.zeros((3, 3))
    mE = np.zeros(3); PE = np.zeros((3, 3))
    for _ in range(steps):
        w = rng.normal(size=(N, 3)) * np.array([sv, sl, sw])
        Xs = np.array([Xs[i] @ ExpSE2(u + w[i]) for i in range(N)])
        F = Ad_SE2(ExpSE2(-u)); G = Jr_SE2(u)
        P = F @ P @ F.T + G @ Q @ G.T; Xbar = Xbar @ ExpSE2(u)
        th = mE[2]; FE = np.array([[1, 0, -d * np.sin(th)], [0, 1, d * np.cos(th)], [0, 0, 1]])
        GE = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        PE = FE @ PE @ FE.T + GE @ Q @ GE.T; mE = mE + [d * np.cos(th), d * np.sin(th), 0]
    return Xs, Xbar, P, mE, PE
Xs, Xbar, P, mE, PE = banana()
tau_s = np.array([LogSE2(T2inv(Xbar) @ Xi) for Xi in Xs])
mL = np.einsum("ij,jk,ik->i", tau_s, np.linalg.inv(P), tau_s)
s = np.array([[Xi[0, 2], Xi[1, 2], np.arctan2(Xi[1, 0], Xi[0, 0])] for Xi in Xs]) - mE
mEu = np.einsum("ij,jk,ik->i", s, np.linalg.inv(PE), s)
print(f"inside predicted 95% region: Lie (tangent) Gaussian {np.mean(mL<=CHI2_3_95)*100:.1f}%, Euclidean (x,y,theta) Gaussian {np.mean(mEu<=CHI2_3_95)*100:.1f}%")
print("P (tangent) =\n", P, "\nPE (x,y,theta) =\n", PE)
print("sample mean x:", Xs[:, 0, 2].mean(), " dead-reckoned x:", Xbar[0, 2], "; heading std (deg):", np.degrees(np.sqrt(P[2, 2])))

# ------------------------------------------------------------------ 3.7a ESKF localisation on SE(2)
print("\n=== ESKF on SE(2) with beacons (Sola et al., sec. V-A) ===")
def eskf_run(seed=5, K=300, dt=0.1, v=1.0, w=0.2, sv=0.1, sw=0.08, sy=0.1, beacons=None):
    rng = np.random.default_rng(seed)
    B = beacons if beacons is not None else np.array([[2.0, 0.0], [2.0, 1.0], [2.0, -1.0]])
    Xt = np.eye(3); Xe = np.eye(3); Xd = np.eye(3); P = np.diag([1e-4] * 3)
    Q = np.diag([(sv * dt) ** 2, (1e-3) ** 2, (sw * dt) ** 2]); Rn = sy**2 * np.eye(2)
    out = dict(true=[Xt], est=[Xe], dr=[Xd], P=[P], nees=[])
    u = np.array([v * dt, 0, w * dt])
    for k in range(K):
        Xt = Xt @ ExpSE2(u)
        um = u + rng.normal(size=3) * np.sqrt(np.diag(Q))          # noisy odometry
        Xd = Xd @ ExpSE2(um)
        F, G = Ad_SE2(ExpSE2(-um)), Jr_SE2(um)
        Xe = Xe @ ExpSE2(um); P = F @ P @ F.T + G @ Q @ G.T
        for b in B:
            yk = (T2inv(Xt) @ np.r_[b, 1])[:2] + rng.normal(size=2) * sy
            e = (T2inv(Xe) @ np.r_[b, 1])[:2]
            Rt = Xe[:2, :2]
            # H = d(X^-1 b)/dX = [ -I , -J R^T (b - t) ]   (right perturbation)
            Hm = np.column_stack([-np.eye(2), -J2 @ (Rt.T @ (b - Xe[:2, 2]))])
            S = Hm @ P @ Hm.T + Rn; Kg = P @ Hm.T @ np.linalg.inv(S)
            dx = Kg @ (yk - e); Xe = Xe @ ExpSE2(dx); P = (np.eye(3) - Kg @ Hm) @ P
        err = LogSE2(T2inv(Xe) @ Xt)
        out["nees"].append(err @ np.linalg.inv(P) @ err)
        out["true"].append(Xt); out["est"].append(Xe); out["dr"].append(Xd); out["P"].append(P)
    return out
o = eskf_run()
eE = [np.linalg.norm(a_[:2, 2] - b_[:2, 2]) for a_, b_ in zip(o["est"], o["true"])]
eD = [np.linalg.norm(a_[:2, 2] - b_[:2, 2]) for a_, b_ in zip(o["dr"], o["true"])]
print(f"final position error: dead reckoning {eD[-1]:.3f} m, ESKF {eE[-1]:.3f} m; mean ESKF error {np.mean(eE):.3f} m; mean NEES {np.mean(o['nees']):.2f} (ideal 3)")
# FD check of the measurement Jacobian
Xc = ExpSE2([0.3, -0.2, 0.7]); bc = np.array([2.0, 1.0])
Hn = num_rjac(lambda A: (T2inv(A) @ np.r_[bc, 1])[:2], Xc, "SE2")
Ha = np.column_stack([-np.eye(2), -J2 @ (Xc[:2, :2].T @ (bc - Xc[:2, 2]))])
print("measurement Jacobian FD check:", np.abs(Hn - Ha).max())
np.savez(os.path.join(HERE, "eskf.npz"), true=np.array(o["true"]), est=np.array(o["est"]), dr=np.array(o["dr"]), P=np.array(o["P"]))

# ------------------------------------------------------------------ 3.7b pose graph on SE(2)
print("\n=== pose-graph optimisation on SE(2) ===")
def pg_problem(seed=2):
    rng = np.random.default_rng(seed)
    steps = [np.array([1.0, 0, 0])] * 3 + [np.array([1.0, 0, np.pi / 2])]
    U = steps * 4                                                   # a 4x4 m square, 16 edges
    Xt = [np.eye(3)]
    for uu in U: Xt.append(Xt[-1] @ ExpSE2(uu))
    sig = np.array([0.03, 0.03, 0.02])
    Z = [(i, i + 1, ExpSE2(LogSE2(T2inv(Xt[i]) @ Xt[i + 1]) + rng.normal(size=3) * sig)) for i in range(len(U))]
    Z.append((0, len(U), ExpSE2(LogSE2(T2inv(Xt[0]) @ Xt[-1]) + rng.normal(size=3) * sig * 0.5)))   # loop closure
    X0 = [np.eye(3)]
    for (i, j, Zij) in Z[:-1]: X0.append(X0[-1] @ Zij)                # odometry initial guess
    return Xt, Z, X0, sig
def residual_jac(Xi, Xj, Zij):
    E = T2inv(Zij) @ T2inv(Xi) @ Xj; r = LogSE2(E)
    Jri = np.linalg.inv(Jr_SE2(r))
    return r, -Jri @ Ad_SE2(T2inv(Xj) @ Xi), Jri
# FD check of the residual Jacobians
Xi_, Xj_, Z_ = ExpSE2([0.2, 0.1, 0.3]), ExpSE2([1.1, 0.3, 0.9]), ExpSE2([0.85, 0.15, 0.55])
r_, Ji_, Jj_ = residual_jac(Xi_, Xj_, Z_)
print("FD check d r/d X_i:", np.abs(num_rjac(lambda A: LogSE2(T2inv(Z_) @ T2inv(A) @ Xj_), Xi_, "SE2") - Ji_).max(),
      " d r/d X_j:", np.abs(num_rjac(lambda B: LogSE2(T2inv(Z_) @ T2inv(Xi_) @ B), Xj_, "SE2") - Jj_).max())
Xt, Z, X0, sig = pg_problem()
Xs_ = [M.copy() for M in X0]; Om = np.diag(1 / sig**2); n = len(Xs_)
def cost(Xs_): return sum(LogSE2(T2inv(Zij) @ T2inv(Xs_[i]) @ Xs_[j]) @ Om @ LogSE2(T2inv(Zij) @ T2inv(Xs_[i]) @ Xs_[j]) for i, j, Zij in Z)
def ate(Xs_): return np.sqrt(np.mean([np.sum((A[:2, 2] - B[:2, 2]) ** 2) for A, B in zip(Xs_, Xt)]))
print(f"initial: cost {cost(Xs_):.1f}, position RMSE {ate(Xs_):.3f} m, end-point gap {np.linalg.norm(Xs_[-1][:2,2]-Xs_[0][:2,2]):.3f} m")
for it in range(6):
    Hh = np.zeros((3 * n, 3 * n)); bb = np.zeros(3 * n)
    Hh[:3, :3] += 1e8 * np.eye(3)                                    # gauge: fix the first pose
    for i, j, Zij in Z:
        r, Ji, Jj = residual_jac(Xs_[i], Xs_[j], Zij)
        for (a_, Ja) in ((i, Ji), (j, Jj)):
            bb[3 * a_:3 * a_ + 3] += Ja.T @ Om @ r
            for (b_, Jb) in ((i, Ji), (j, Jj)):
                Hh[3 * a_:3 * a_ + 3, 3 * b_:3 * b_ + 3] += Ja.T @ Om @ Jb
    dx = -np.linalg.solve(Hh, bb)
    Xs_ = [Xs_[k] @ ExpSE2(dx[3 * k:3 * k + 3]) for k in range(n)]
    print(f" iter {it+1}: cost {cost(Xs_):.3f}, |dx| {np.linalg.norm(dx):.2e}, position RMSE {ate(Xs_):.3f} m")
np.savez(os.path.join(HERE, "posegraph.npz"), true=np.array(Xt), init=np.array(X0), opt=np.array(Xs_))

print("\n=== banana: coverage vs heading noise ===")
for sw_ in [0.03, 0.06, 0.10]:
    Xs2, Xb2, P2, mE2, PE2 = banana(N=4000, sw=sw_)
    ts = np.array([LogSE2(T2inv(Xb2) @ Xi) for Xi in Xs2]); mL2 = np.einsum("ij,jk,ik->i", ts, np.linalg.inv(P2), ts)
    s2 = np.array([[Xi[0, 2], Xi[1, 2], np.arctan2(Xi[1, 0], Xi[0, 0])] for Xi in Xs2]) - mE2
    mE_ = np.einsum("ij,jk,ik->i", s2, np.linalg.inv(PE2), s2)
    print(f"sigma_w/step={sw_}: final heading std {np.degrees(np.sqrt(P2[2,2])):.1f} deg -> inside 95%: Lie {np.mean(mL2<=CHI2_3_95)*100:.1f}%, Euclidean {np.mean(mE_<=CHI2_3_95)*100:.1f}%")
