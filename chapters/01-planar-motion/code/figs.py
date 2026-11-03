"""Figures for chapter 1 -> ../figs/*.pdf.   Run: make figs CH=01-planar-motion"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from rigid import *
from style import *

OUT = os.path.join(HERE, "..", "figs") + os.sep
TEAL = "#0b7f6d"
Tx = lambda d: T2(0.0, [d, 0.0])
TR = lambda a: T2(a, [0.0, 0.0])


def robot(ax, M, col=TEAL, s=0.16, alpha=1.0, z=5, fill="#e8f4f1"):
    """Small differential-drive glyph at pose M (3x3), heading arrow along body x."""
    th = np.arctan2(M[1, 0], M[0, 0]); x, y = M[0, 2], M[1, 2]
    tr = Affine2D().rotate(th).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-0.6 * s, -0.45 * s), 1.2 * s, 0.9 * s, fc=fill, ec=col, lw=1.0, transform=tr, alpha=alpha, zorder=z))
    for sy in (-1, 1):
        ax.add_patch(Rectangle((-0.35 * s, sy * 0.45 * s - 0.09 * s), 0.7 * s, 0.18 * s, fc="#aaaaaa", ec="none", transform=tr, alpha=alpha, zorder=z))
    ax.annotate("", xy=(x + 0.9 * s * np.cos(th), y + 0.9 * s * np.sin(th)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=8), zorder=z + 1, alpha=alpha)
    ax.plot(x, y, "o", color=col, ms=2.2, zorder=z + 1, alpha=alpha)


def arc_path(S, n=200, lam=1.0):
    return np.array([ExpSE2(np.asarray(S) * l)[:2, 2] for l in np.linspace(0, lam, n)])


def clean(ax, lim=None):
    ax.set_aspect("equal")
    if lim: ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)


# --------------------------------------------------------------------------- F1
def f_step_vs_arc():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    # (a) one big step
    ax = axs[0]; r, d = 1.0, 0.6
    P = arc_path([r, 0, 1], lam=d)
    ax.plot(P[:, 0], P[:, 1], color=CZ, lw=1.8, label="exact arc $\\exp(\\delta\\theta\\,G)$")
    ax.plot([0, r * d], [0, 0], color=ORA, lw=1.8, label="move $r\\delta\\theta$, then turn")
    ax.plot(0, r, "o", color=DK, ms=3); ax.text(-0.13, r - 0.03, "$C$", fontsize=8)
    ax.plot([0, 0], [0, r], ":", color=GRY, lw=0.8); ax.plot([0, P[-1, 0]], [r, P[-1, 1]], ":", color=GRY, lw=0.8)
    robot(ax, np.eye(3), col=GRY, fill="#f2f2f2", s=0.13)
    robot(ax, ExpSE2([r * d, 0, d]), col=CZ, s=0.13, fill="#e3edf7")
    robot(ax, Tx(r * d) @ TR(d), col=ORA, s=0.13, fill="#fbeedd")
    ax.annotate("", xy=(P[-1, 0], P[-1, 1]), xytext=(r * d, 0), arrowprops=dict(arrowstyle="<->", color=CX, lw=0.9))
    ax.text(0.7, -0.22, "gap $\\approx r\\,\\delta\\theta^2/2$", color=CX, fontsize=7.5)
    clean(ax, (-0.2, 1.0, -0.25, 1.15)); ax.set_title("(a) one step, $\\delta\\theta=0.6$", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.12, 0.9))
    # (b) n steps
    ax = axs[1]; th = np.pi / 2
    P = arc_path([r, 0, 1], lam=th); ax.plot(P[:, 0], P[:, 1], color=CZ, lw=2.2, label="$\\exp(\\theta G)$", zorder=4)
    for n, c in zip([1, 2, 4, 8], ["#f0b27a", ORA, "#c0611a", "#7a3b0d"]):
        M = np.eye(3); pts = [M[:2, 2]]
        for _ in range(n):
            M = M @ Tx(r * th / n); pts.append(M[:2, 2]); M = M @ TR(th / n)
        pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], "-o", color=c, lw=1.0, ms=2, label=f"$n={n}$")
    ax.plot(0, r, "o", color=DK, ms=3); ax.text(-0.12, r + 0.04, "$C$", fontsize=8)
    clean(ax, (-0.15, 1.75, -0.15, 1.35)); ax.set_title("(b) $(T_x(r\\theta/n)\\,T_R(\\theta/n))^n$", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1)
    # (c) error vs n
    ax = axs[2]; ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]); Tt = ExpSE2([r * th, 0, th])
    G = hat2([r, 0, 1]); e1, e2, e3 = [], [], []
    for n in ns:
        dd = th / n
        e1.append(np.linalg.norm(np.linalg.matrix_power(np.eye(3) + dd * G, n)[:2, 2] - Tt[:2, 2]))
        e2.append(np.linalg.norm(np.linalg.matrix_power(Tx(r * dd) @ TR(dd), n)[:2, 2] - Tt[:2, 2]))
        e3.append(np.linalg.norm(np.linalg.matrix_power(TR(dd / 2) @ Tx(r * dd) @ TR(dd / 2), n)[:2, 2] - Tt[:2, 2]))
    ax.loglog(ns, e1, "s-", color=GRY, ms=3, lw=1, label="$I+\\delta\\theta G$")
    ax.loglog(ns, e2, "o-", color=ORA, ms=3, lw=1, label="move, turn")
    ax.loglog(ns, e3, "^-", color=CY, ms=3, lw=1, label="half-turn, move, half-turn")
    ax.loglog(ns, 1.1 / ns, ":", color=DK, lw=0.8); ax.text(150, 0.02, "$\\propto 1/n$", fontsize=7.5)
    ax.loglog(ns, 0.16 / ns**2, ":", color=DK, lw=0.8); ax.text(2.2, 2e-4, "$\\propto 1/n^2$", fontsize=7.5)
    ax.set_xlabel("number of steps $n$", fontsize=8); ax.set_ylabel("end-point error [m]", fontsize=8)
    ax.set_title("(c) convergence to $\\exp(\\theta G)$", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left"); ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "step_vs_arc.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F2
def field(ax, S, lim, title, show_C=True, path_lam=None):
    v, w = np.array(S[:2]), S[2]
    g = np.linspace(lim[0] + 0.15, lim[1] - 0.15, 9); h = np.linspace(lim[2] + 0.15, lim[3] - 0.15, 9)
    X, Y = np.meshgrid(g, h)
    U = v[0] - w * Y; V = v[1] + w * X
    ax.quiver(X, Y, U, V, color=GRY, scale=9, width=0.006, headwidth=4)
    if w != 0 and show_C:
        C = J2 @ v / w; ax.plot(*C, "o", color=CX, ms=4, zorder=6); ax.text(C[0] + 0.07, C[1] + 0.07, "$C=Jv/\\omega$", color=CX, fontsize=7.5)
        for rr in (0.4, 0.8, 1.2):
            t = np.linspace(0, 2 * np.pi, 100); ax.plot(C[0] + rr * np.cos(t), C[1] + rr * np.sin(t), color=CX, lw=0.4, alpha=0.4)
    if path_lam:
        P = arc_path(S, lam=path_lam); ax.plot(P[:, 0], P[:, 1], color=TEAL, lw=1.8, zorder=5)
        robot(ax, np.eye(3), col=GRY, fill="#f2f2f2", s=0.2); robot(ax, ExpSE2(np.asarray(S) * path_lam), s=0.2)
    ax.annotate("", xy=(v[0] * 0.6, v[1] * 0.6), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=PUR, lw=1.4), zorder=7)
    ax.text(v[0] * 0.6 + 0.03, v[1] * 0.6 - 0.18, "$v$", color=PUR, fontsize=8)
    clean(ax, lim); ax.set_title(title, fontsize=8.5)


def f_twist_field():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.4))
    field(axs[0], (1, 0, 1), (-1.3, 1.3, -0.7, 1.9), "(a) $S=(r,0,1)$, $r=1$: left turn", path_lam=np.pi / 2)
    field(axs[1], (0.3, 0.2, 0.5), (-1.3, 1.3, -0.7, 1.9), "(b) $S=(0.3,0.2,0.5)$", path_lam=2.4)
    field(axs[2], (1, 0, 0), (-1.3, 1.3, -0.7, 1.9), "(c) $S=(1,0,0)$: pure translation", path_lam=1.0)
    fig.tight_layout(w_pad=0.4); fig.savefig(OUT + "twist_field.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F3
def f_log_branch():
    fig, ax = plt.subplots(figsize=(HALF_W + 0.3, 2.6))
    r, th = 1.0, 1.5 * np.pi
    P = arc_path([r, 0, 1], lam=th); ax.plot(P[:, 0], P[:, 1], color=CZ, lw=1.8, label="driven: $\\exp(\\frac{3\\pi}{2}[(1,0,1)])$")
    S = LogSE2(ExpSE2([r * th, 0, th]))
    Q = arc_path(S, lam=1.0); ax.plot(Q[:, 0], Q[:, 1], color=ORA, lw=1.8, ls="--", label="principal log: $(-\\frac{\\pi}{2},0,-\\frac{\\pi}{2})$")
    for l in (0.3, 0.6):
        robot(ax, ExpSE2(S * l), col=ORA, s=0.17, alpha=0.5, fill="#fbeedd")
    robot(ax, np.eye(3), col=GRY, fill="#f2f2f2", s=0.17); robot(ax, ExpSE2([r * th, 0, th]), s=0.17)
    ax.plot(0, 1, "o", color=DK, ms=3); ax.text(0.06, 1.05, "$C$", fontsize=8)
    ax.text(0.05, -0.35, "start", fontsize=7.5, color=GRY); ax.text(-1.32, 0.65, "end", fontsize=7.5, color=TEAL)
    clean(ax, (-1.5, 1.5, -0.5, 2.4)); ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=1)
    fig.tight_layout(); fig.savefig(OUT + "log_branch.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F4
def odom(fr, Tend=60.0):
    h = 1 / fr; n_ = int(Tend * fr); t_ = np.arange(n_) * h
    w_ = 0.8 * np.cos(2 * np.pi * (t_ + h / 2) / 8.0); v_ = 0.6 + 0 * t_
    Tq = np.eye(3); pe = np.zeros(3); pr = np.zeros(3); X = [np.zeros(3)]; E = [pe]; K = [pr]
    for k in range(n_):
        d, a = v_[k] * h, w_[k] * h
        Tq = Tq @ ExpSE2([d, 0, a]); X.append(np.array([Tq[0, 2], Tq[1, 2], 0]))
        pe = pe + [d * np.cos(pe[2]), d * np.sin(pe[2]), a]; E.append(pe)
        pr = pr + [d * np.cos(pr[2] + a / 2), d * np.sin(pr[2] + a / 2), a]; K.append(pr)
    return np.r_[t_, Tend], np.array(X), np.array(E), np.array(K)


def f_odometry():
    fig = plt.figure(figsize=(FULL_W, 3.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1], width_ratios=[1, 1.15], hspace=0.55, wspace=0.3)
    t, X, E, K = odom(2.0)
    h = 0.5; tt = np.arange(len(t) - 1) * h; w_ = 0.8 * np.cos(2 * np.pi * (tt + h / 2) / 8.0)
    M = np.eye(3); Xf = []
    for k in range(len(tt)):   # the true continuous motion: an exact arc between samples
        for l in np.linspace(0, 1, 12):
            Xf.append((M @ ExpSE2([0.6 * h * l, 0, w_[k] * h * l]))[:2, 2])
        M = M @ ExpSE2([0.6 * h, 0, w_[k] * h])
    Xf = np.array(Xf)
    def draw(ax, ms):
        ax.plot(Xf[:, 0], Xf[:, 1], color=CZ, lw=3.0, alpha=0.3, label="true path (continuous)")
        ax.plot(X[:, 0], X[:, 1], "o", color=CZ, ms=ms, label="exact exp, at samples")
        ax.plot(E[:, 0], E[:, 1], "-", color=CX, lw=1.0, label="Euler: move, then turn")
        ax.plot(K[:, 0], K[:, 1], "--", color=CY, lw=1.0, label="RK2: midpoint heading")
        ax.set_aspect("equal"); ax.tick_params(labelsize=7.5); ax.grid(alpha=0.25, lw=0.4)
    ax = fig.add_subplot(gs[0, :]); draw(ax, 1.6)
    ax.set_xlim(-0.5, 27.8); ax.set_ylim(-1.2, 1.2); ax.set_xlabel("$x$ [m]", fontsize=8, labelpad=1); ax.set_ylabel("$y$ [m]", fontsize=8)
    ax.add_patch(Rectangle((23.6, -1.1), 4.0, 2.2, fill=False, ec=DK, lw=0.7, ls=":"))
    ax.set_title("(a) slalom at 0.6 m/s, 36 m, wheel encoders read at 2 Hz", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.55))
    ax = fig.add_subplot(gs[1, 0]); draw(ax, 3)
    ax.set_xlim(23.6, 27.6); ax.set_ylim(-1.1, 1.1); ax.set_title("(b) zoom on the last 4 m", fontsize=8.5)
    ax = fig.add_subplot(gs[1, 1])
    for fr, ls in [(10.0, "-"), (2.0, "--")]:
        t, X2, E2, K2 = odom(fr)
        ax.semilogy(t, np.linalg.norm(E2[:, :2] - X2[:, :2], axis=1) * 100 + 1e-9, ls, color=CX, lw=1.1, label=f"Euler, {fr:.0f} Hz")
        ax.semilogy(t, np.linalg.norm(K2[:, :2] - X2[:, :2], axis=1) * 100 + 1e-9, ls, color=CY, lw=1.1, label=f"RK2, {fr:.0f} Hz")
    ax.set_ylim(1e-3, 100); ax.set_xlabel("time [s]", fontsize=8); ax.set_ylabel("error vs. exact [cm]", fontsize=8)
    ax.set_title("(c) error caused by the integrator alone", fontsize=8.5); ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, lw=0.4, which="both"); ax.tick_params(labelsize=7.5)
    fig.savefig(OUT + "odometry.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F5
def f_bracket():
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    A, B = [1, 0, 0], [0, 0, 1]; eps = 0.35
    steps = [(A, eps, "1: forward"), (B, eps, "2: turn left"), (A, -eps, "3: backward"), (B, -eps, "4: turn right")]
    ax = axs[0]; M = np.eye(3); robot(ax, M, col=GRY, fill="#f2f2f2", s=0.11)
    cols = [CZ, ORA, PUR, CY]
    for (S, a, lab), c in zip(steps, cols):
        P = np.array([(M @ ExpSE2(np.asarray(S, float) * a * l))[:2, 2] for l in np.linspace(0, 1, 30)])
        ax.plot(P[:, 0], P[:, 1], color=c, lw=1.6, label=lab)
        M = M @ ExpSE2(np.asarray(S, float) * a)
        robot(ax, M, col=c, s=0.11, alpha=0.55, fill="white")
    robot(ax, M, col=CX, s=0.11, fill="#f7e1de")
    ax.annotate("", xy=(M[0, 2], M[1, 2]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CX, lw=1.2), zorder=9)
    ax.text(-0.22, -0.3, "net motion $\\approx \\varepsilon^2[A,B]$: sideways", color=CX, fontsize=7.5)
    clean(ax, (-0.25, 0.6, -0.36, 0.22)); ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"(a) one cycle, $\\varepsilon={eps}$ (m and rad)", fontsize=8.5)
    ax = axs[1]; eps = 0.2; M = np.eye(3); pts = [M[:2, 2]]
    for k in range(10):
        for S, a, _ in [(A, eps, ""), (B, eps, ""), (A, -eps, ""), (B, -eps, "")]:
            for l in np.linspace(0, 1, 8)[1:]:
                pts.append((M @ ExpSE2(np.asarray(S, float) * a * l))[:2, 2])
            M = M @ ExpSE2(np.asarray(S, float) * a)
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color=CZ, lw=0.8)
    robot(ax, np.eye(3), col=GRY, fill="#f2f2f2", s=0.1); robot(ax, M, col=CX, s=0.1, fill="#f7e1de")
    ax.text(0.32, -0.25, f"10 cycles, $\\varepsilon=0.2$:\nnet $({M[0,2]:.2f},\\,{M[1,2]:.2f})$ m\nheading $0^\\circ$", fontsize=7.5)
    clean(ax, (-0.3, 0.9, -0.5, 0.25)); ax.set_title("(b) repeating it walks the robot sideways", fontsize=8.5)
    fig.tight_layout(w_pad=0.5); fig.savefig(OUT + "bracket.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in [f_step_vs_arc, f_twist_field, f_log_branch, f_odometry, f_bracket]:
        f(); print("ok", f.__name__)
