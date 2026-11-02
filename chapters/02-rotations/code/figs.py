import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))  # repo-level shared library
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import expm
from rigid import *

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
CX, CY, CZ = "#c0392b", "#27884a", "#2c6fbb"
ORA, PUR, GRY, DK = "#e67e22", "#8e44ad", "#9a9a9a", "#333333"
OUT = os.path.join(HERE, "..", "figs") + os.sep


def ax3d(fig, pos, lim=1.1, elev=22, azim=35, zoom=1.45):
    ax = fig.add_subplot(*pos, projection="3d") if isinstance(pos, tuple) else fig.add_subplot(pos, projection="3d")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=zoom); ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return ax


def arrow(ax, o, d, c, lw=1.6, ls="-", alr=0.12, label=None, lpos=1.12, fs=9, **kw):
    o = np.asarray(o, float); d = np.asarray(d, float)
    ax.quiver(*o, *d, color=c, lw=lw, linestyle=ls, arrow_length_ratio=alr, **kw)
    if label:
        ax.text(*(o + lpos * d), label, color=c, fontsize=fs, ha="center", va="center")


def frame(ax, R=np.eye(3), o=(0, 0, 0), s=1.0, labels=("x", "y", "z"), lw=1.6, alpha=1.0, ls="-", fs=9, cols=(CX, CY, CZ)):
    for i in range(3):
        arrow(ax, o, s * R[:, i], cols[i], lw=lw, ls=ls, label=labels[i] if labels else None, alpha=alpha, fs=fs)


# ---------------------------------------------------------------- F1 frames
def f_frames():
    R = zyx(50 * deg, -25 * deg, 30 * deg)
    fig = plt.figure(figsize=(6.6, 3.1))
    ax = ax3d(fig, (1, 2, 1), lim=0.85, elev=20, azim=20, zoom=1.3)
    frame(ax, np.eye(3), labels=(r"$a_x$", r"$a_y$", r"$a_z$"), cols=(GRY, GRY, GRY), lw=1.2)
    frame(ax, R, labels=(r"$b_x$", r"$b_y$", r"$b_z$"), lw=2.0)
    ax.set_title(r"(a) orientation of $\{B\}$ relative to $\{A\}$")
    ax2 = ax3d(fig, (1, 2, 2), lim=0.75, elev=22, azim=20, zoom=1.3)
    frame(ax2, np.eye(3), labels=(r"$a_x$", r"$a_y$", r"$a_z$"), cols=(GRY, GRY, GRY), lw=1.0)
    b = R[:, 0]
    arrow(ax2, (0, 0, 0), b, CX, lw=2.2, label=r"$b_x$")
    p0 = np.zeros(3); p1 = np.array([b[0], 0, 0]); p2 = np.array([b[0], b[1], 0]); p3 = b
    for a_, b_, c_, t_ in [(p0, p1, DK, r"$r_{11}=a_x\!\cdot b_x$"), (p1, p2, DK, r"$r_{21}=a_y\!\cdot b_x$"), (p2, p3, DK, r"$r_{31}=a_z\!\cdot b_x$")]:
        ax2.plot(*np.c_[a_, b_], color=ORA, lw=2.2)
    ax2.text(*(p1 * 0.5 + np.array([0, -0.18, -0.12])), r"$r_{11}$", color=ORA)
    ax2.text(*((p1 + p2) / 2 + np.array([0.12, 0, -0.12])), r"$r_{21}$", color=ORA)
    ax2.text(*((p2 + p3) / 2 + np.array([0.1, 0.05, 0])), r"$r_{31}$", color=ORA)
    ax2.text2D(0.02, 0.02, r"$b_x = r_{11}a_x + r_{21}a_y + r_{31}a_z$" "\n" r"$\Rightarrow$ column 1 of $R_{AB}$ = $b_x$ written in $\{A\}$",
               transform=ax2.transAxes, fontsize=8.5)
    fig.subplots_adjust(wspace=0.05)
    ax2.set_title(r"(b) column 1 of $R_{AB}$ = $b_x$ in $\{A\}$-coordinates")
    fig.savefig(OUT + "frames.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F2 active/passive
def f_active_passive():
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3))
    th = 35 * deg
    p = np.array([0.9, 0.35])
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlim(-0.8, 1.3); ax.set_ylim(-0.35, 1.25); ax.axis("off")
    # (a) derivation of Rz: images of basis vectors
    ax = axs[0]
    for v, c, l in [((1, 0), GRY, r"$e_1$"), ((0, 1), GRY, r"$e_2$")]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))
        ax.text(v[0] + 0.04, v[1] + 0.04, l, color=c)
    e1p, e2p = R2[:, 0], R2[:, 1]
    ax.annotate("", xy=e1p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CX, lw=1.8))
    ax.annotate("", xy=e2p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CY, lw=1.8))
    ax.text(e1p[0] + 0.03, e1p[1] + 0.03, r"$(\cos\theta,\ \sin\theta)$", color=CX, fontsize=8)
    ax.text(e2p[0] - 0.2, e2p[1] + 0.08, r"$(-\sin\theta,\ \cos\theta)$", color=CY, fontsize=8)
    ax.plot([e1p[0], e1p[0]], [0, e1p[1]], ":", color=CX, lw=0.8)
    a = np.linspace(0, th, 30); ax.plot(0.35 * np.cos(a), 0.35 * np.sin(a), color=DK, lw=0.8)
    ax.text(0.38, 0.08, r"$\theta$")
    ax.set_title("(a) where the basis vectors go\n" r"$\Rightarrow$ columns of $R_z(\theta)$", fontsize=8.5)
    # (b) active
    ax = axs[1]
    for v in [(1, 0), (0, 1)]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GRY, lw=1.0))
    ax.text(1.03, -0.02, r"$x$", color=GRY); ax.text(-0.02, 1.04, r"$y$", color=GRY)
    pp = R2 @ p
    ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CZ, lw=1.8))
    ax.annotate("", xy=pp, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORA, lw=1.8))
    ax.text(p[0] + 0.03, p[1] - 0.09, r"$p$", color=CZ); ax.text(pp[0] + 0.03, pp[1], r"$p'=Rp$", color=ORA)
    a0 = np.arctan2(p[1], p[0]); a = np.linspace(a0, a0 + th, 30)
    ax.plot(0.55 * np.cos(a), 0.55 * np.sin(a), color=DK, lw=0.8); ax.text(0.5, 0.42, r"$\theta$")
    ax.set_title("(b) active: the vector moves,\nthe frame stays", fontsize=8.5)
    # (c) passive
    ax = axs[2]
    for v in [(1, 0), (0, 1)]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GRY, lw=1.0))
    ax.text(1.03, -0.02, r"$x$", color=GRY); ax.text(-0.02, 1.04, r"$y$", color=GRY)
    for i, (c, l) in enumerate([(CX, r"$x'$"), (CY, r"$y'$")]):
        v = R2[:, i]; ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        ax.text(v[0] + 0.03, v[1] + 0.03, l, color=c)
    ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=CZ, lw=1.8))
    ax.text(p[0] + 0.03, p[1] - 0.09, r"$p$", color=CZ)
    proj = (R2[:, 0] @ p) * R2[:, 0]
    ax.plot([proj[0], p[0]], [proj[1], p[1]], ":", color=CX, lw=0.9)
    ax.text(-0.7, -0.3, r"coords of $p$ in new frame: $R^{\top}p$", fontsize=8.5)
    ax.set_title("(c) passive: the vector stays,\nthe frame moves", fontsize=8.5)
    fig.savefig(OUT + "active_passive.pdf"); plt.close(fig)


# ---------------------------------------------------------------- box helper
def box_faces(R, dims=(0.55, 0.32, 0.16)):
    a, b, c = dims
    V = np.array([[sx * a, sy * b, sz * c] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    idx = lambda sx, sy, sz: ((sx > 0) * 4 + (sy > 0) * 2 + (sz > 0))
    faces, cols = [], []
    spec = [((1, 0, 0), ORA), ((-1, 0, 0), "#dddddd"), ((0, 1, 0), "#7fbf8f"), ((0, -1, 0), "#dddddd"),
            ((0, 0, 1), "#7fa8d9"), ((0, 0, -1), "#dddddd")]
    for n, col in spec:
        k = np.argmax(np.abs(n)); s = n[k]
        others = [i for i in range(3) if i != k]
        quad = []
        for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            sg = [0, 0, 0]; sg[k] = s; sg[others[0]] = u; sg[others[1]] = v
            quad.append(V[idx(*sg)])
        faces.append((R @ np.array(quad).T).T); cols.append(col)
    return faces, cols


def draw_box(ax, R, alpha=0.9):
    f, c = box_faces(R)
    ax.add_collection3d(Poly3DCollection(f, facecolors=c, edgecolors="#555555", linewidths=0.5, alpha=alpha))
    frame(ax, R, s=0.75, labels=None, lw=1.3)


# ---------------------------------------------------------------- F3 noncommutativity
def f_noncommute():
    fig = plt.figure(figsize=(6.6, 4.1))
    seqs = [("x", Rx(np.pi / 2), "z", Rz(np.pi / 2)), ("z", Rz(np.pi / 2), "x", Rx(np.pi / 2))]
    for r, (n1, A, n2, B) in enumerate(seqs):
        Rs = [np.eye(3), A, B @ A]
        titles = ["start", rf"after $R_{n1}(90^\circ)$", rf"then $R_{n2}(90^\circ)$" "\n" rf"total $R_{n2}R_{n1}$"]
        for cidx in range(3):
            ax = ax3d(fig, (2, 3, 3 * r + cidx + 1), lim=0.8, elev=24, azim=40, zoom=1.3)
            for i, (c, l) in enumerate(zip((CX, CY, CZ), ("X", "Y", "Z"))):
                e = np.eye(3)[i] * 0.95
                ax.plot([0, e[0]], [0, e[1]], [0, e[2]], color=c, lw=0.6, ls=":", alpha=0.7)
                ax.text(*(e * 1.08), l, color=c, fontsize=7, alpha=0.8)
            draw_box(ax, Rs[cidx])
            ax.set_title(titles[cidx], fontsize=8.5)
    fig.text(0.0, 0.73, "order 1", rotation=90, fontsize=9); fig.text(0.0, 0.26, "order 2", rotation=90, fontsize=9)
    fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.savefig(OUT + "noncommute.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F4 ZYX sequence
def f_zyx_seq():
    psi, th, ph = 40 * deg, 30 * deg, 35 * deg
    Rs = [np.eye(3), Rz(psi), Rz(psi) @ Ry(th), Rz(psi) @ Ry(th) @ Rx(ph)]
    titles = ["(0) start: body = world", r"(1) yaw $\psi$ about $z$", r"(2) pitch $\theta$ about new $y'$",
              r"(3) roll $\phi$ about newest $x''$"]
    axes_of_rot = [None, (Rs[0], 2), (Rs[1], 1), (Rs[2], 0)]
    fig = plt.figure(figsize=(6.6, 1.95))
    for k in range(4):
        ax = ax3d(fig, (1, 4, k + 1), lim=0.9, elev=20, azim=35)
        if k > 0:
            frame(ax, Rs[k - 1], s=0.85, labels=None, lw=0.9, ls="--", alpha=0.45)
            Rp, i = axes_of_rot[k]
            a = Rp[:, i]
            ax.plot(*np.c_[-1.0 * a, 1.0 * a], color=PUR, lw=2.6, alpha=0.35)
        frame(ax, Rs[k], s=0.85, labels=("x", "y", "z"), lw=1.8, fs=8)
        ax.set_title(titles[k], fontsize=8.2)
    fig.subplots_adjust(wspace=0.0)
    fig.savefig(OUT + "zyx_seq.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F5 gimbal
def ring(ax, u, v, r, c, lw=2.4, alpha=1.0):
    t = np.linspace(0, 2 * np.pi, 200)
    P = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=c, lw=lw, alpha=alpha)


def gimbal(ax, psi, th, ph, title):
    z = np.array([0, 0, 1.])
    ap = Rz(psi) @ np.array([0, 1, 0.])
    ar = Rz(psi) @ Ry(th) @ np.array([1, 0, 0.])
    R = zyx(psi, th, ph)
    def perp(a, b):
        b = b - (a @ b) * a
        return b / np.linalg.norm(b)
    ring(ax, z, ap, 1.0, CZ)
    ring(ax, ap, perp(ap, ar), 0.8, CY)
    ring(ax, ar, perp(ar, R[:, 1]), 0.6, CX)
    for a, c, L, l in [(z, CZ, 1.35, "yaw axis"), (ap, CY, 1.15, "pitch axis"), (ar, CX, 0.95, "roll axis")]:
        ax.plot(*np.c_[-L * a, L * a], color=c, lw=1.0, ls="--")
    arrow(ax, (0, 0, 0), 0.5 * R[:, 0], DK, lw=2.4)
    ax.text(*(0.62 * R[:, 0]), "nose", fontsize=7.5, color=DK)
    ax.set_title(title, fontsize=8.5)


def f_gimbal():
    fig = plt.figure(figsize=(6.6, 2.6))
    ax = ax3d(fig, (1, 3, 1), lim=1.0, elev=22, azim=-40, zoom=1.35)
    gimbal(ax, 30 * deg, 25 * deg, 20 * deg, r"(a) $\theta=25^\circ$: three independent axes")
    ax.text2D(0.0, 0.0, "blue: yaw   green: pitch   red: roll", transform=ax.transAxes, fontsize=7.5)
    ax = ax3d(fig, (1, 3, 2), lim=1.0, elev=22, azim=-40, zoom=1.35)
    gimbal(ax, 30 * deg, 90 * deg, 20 * deg, r"(b) $\theta=90^\circ$: roll axis $\parallel$ yaw axis")
    ax.text2D(0.0, 0.0, "outer & middle rings coplanar:\nno ring can turn the nose sideways", transform=ax.transAxes, fontsize=7.5)
    ax = fig.add_subplot(1, 3, 3)
    t = np.linspace(0, 89.4, 400) * deg
    ax.semilogy(t / deg, 1 / np.cos(t), color=CZ, lw=1.6, label=r"$|\dot\psi|=1/\cos\theta$")
    ax.semilogy(t / deg, np.abs(np.tan(t)) + 1e-3, color=CX, lw=1.6, ls="--", label=r"$|\dot\phi|=\tan\theta$")
    ax.axvline(90, color=GRY, lw=0.8)
    ax.set_xlabel(r"pitch $\theta$ [deg]"); ax.set_ylabel("Euler rate [rad/s]")
    ax.set_title(r"(c) Euler rates for $\omega_b=(0,0,1)$ rad/s", fontsize=8.5)
    ax.legend(loc="upper left", frameon=False); ax.set_xlim(0, 92); ax.set_ylim(1e-2, 300)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(OUT + "gimbal.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F6 Rodrigues
def f_rodrigues():
    th = 70 * deg
    w = np.array([0, 0, 1.]); v = np.array([0.8, 0.0, 0.6])
    vpar = (w @ v) * w; vperp = v - vpar; wxv = np.cross(w, v)
    vp = vpar + np.cos(th) * vperp + np.sin(th) * wxv
    fig = plt.figure(figsize=(6.6, 2.9))
    ax = ax3d(fig, (1, 2, 1), lim=0.9, elev=24, azim=-58, zoom=1.6)
    ax.set_zlim(-0.1, 1.3); ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85)
    arrow(ax, (0, 0, 0), 1.2 * w, DK, lw=1.6, label=r"$\hat\omega$", lpos=1.06)
    arrow(ax, (0, 0, 0), v, CZ, lw=2.0, label=r"$v$")
    arrow(ax, (0, 0, 0), vp, ORA, lw=2.0, label=r"$v'$")
    arrow(ax, (0, 0, 0), vpar, PUR, lw=1.8, label=r"$v_\parallel$", lpos=0.55)
    ax.text(*(vpar * 0.5 + np.array([-0.32, 0, 0])), r"$v_\parallel$", color=PUR)
    arrow(ax, vpar, vperp, CX, lw=1.5, label=r"$v_\perp$", lpos=0.6)
    arrow(ax, vpar, wxv, CY, lw=1.5, label=r"$\hat\omega\times v$", lpos=1.25)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.8 * np.cos(t), 0.8 * np.sin(t), 0.6 + 0 * t, color=GRY, lw=0.8)
    a = np.linspace(0, th, 40)
    ax.plot(0.35 * np.cos(a), 0.35 * np.sin(a), 0.6 + 0 * a, color=DK, lw=1.0)
    ax.text(0.4 * np.cos(th / 2), 0.4 * np.sin(th / 2), 0.66, r"$\theta$")
    ax.set_title("(a) only the perpendicular part rotates;\nit sweeps a circle around the axis", fontsize=8.5)
    ax2 = fig.add_subplot(1, 2, 2); ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_xlim(-0.45, 1.0); ax2.set_ylim(-0.25, 0.98)
    tt = np.linspace(0, 2 * np.pi, 200); ax2.plot(0.8 * np.cos(tt), 0.8 * np.sin(tt), color=GRY, lw=0.6)
    P = lambda x, y, c, l, dx=0.02, dy=0.02: (ax2.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8)),
                                               ax2.text(x + dx, y + dy, l, color=c))
    P(0.8, 0, CX, r"$v_\perp$", 0.02, -0.08)
    P(0, 0.8, CY, r"$\hat\omega\times v$ (same length as $v_\perp$, $90^\circ$ ahead)", -0.05, 0.05)
    P(0.8 * np.cos(th), 0.8 * np.sin(th), ORA, r"$v'_\perp$")
    ax2.plot([0.8 * np.cos(th)] * 2, [0, 0.8 * np.sin(th)], ":", color=DK, lw=0.9)
    ax2.plot([0, 0.8 * np.cos(th)], [0.8 * np.sin(th)] * 2, ":", color=DK, lw=0.9)
    ax2.text(0.8 * np.cos(th) / 2 - 0.12, -0.08, r"$\cos\theta\,v_\perp$", fontsize=8.5)
    ax2.text(-0.02, 0.8 * np.sin(th) / 2, r"$\sin\theta\,(\hat\omega\times v)$", fontsize=8.5, ha="right")
    a = np.linspace(0, th, 30); ax2.plot(0.25 * np.cos(a), 0.25 * np.sin(a), color=DK, lw=0.8)
    ax2.text(0.27, 0.1, r"$\theta$")
    ax2.set_title(r"(b) view down $\hat\omega$: a 2-D rotation inside the plane $\perp\hat\omega$", fontsize=8.5)
    ax2.text(-0.4, -0.22, r"$v'=v_\parallel+\cos\theta\,v_\perp+\sin\theta\,(\hat\omega\times v)$", fontsize=9)
    fig.savefig(OUT + "rodrigues.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F7 manifold
def f_manifold():
    fig = plt.figure(figsize=(3.3, 3.0))
    ax = ax3d(fig, 111, lim=0.95, elev=16, azim=-62, zoom=1.15)
    ax.set_zlim(-0.6, 1.3)
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:22j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GRY, lw=0.3, alpha=0.45)
    n = np.array([0, 0, 1.])
    g = np.linspace(-0.75, 0.75, 2)
    X, Y = np.meshgrid(g, g); ax.plot_surface(X, Y, 1 + 0 * X, color=CZ, alpha=0.15)
    d = np.array([0.95, 0.0, 0])
    arrow(ax, n, d, CX, lw=2.0)
    ax.text(*(n + d + np.array([0.05, 0, 0.3])), "$R+\\delta$\n(leaves the group)", color=CX, fontsize=8, ha="center")
    L = np.linalg.norm(d); dh = d / L
    s_ = np.linspace(0, L, 50)
    G = np.outer(np.cos(s_), n) + np.outer(np.sin(s_), dh)
    ax.plot(G[:, 0], G[:, 1], G[:, 2], color=ORA, lw=2.6)
    ax.scatter(*G[-1], color=ORA, s=18); ax.text(*(G[-1] + np.array([0.12, 0.0, -0.08])), r"$R\,\mathrm{Exp}(\delta)$", color=ORA)
    ax.scatter(*n, color=DK, s=18); ax.text(-0.18, 0, 1.1, r"$R$", color=DK)
    ax.text2D(0.0, 0.79, "tangent plane\n" r"$\cong\mathfrak{so}(3)\cong\mathbb{R}^3$", color=CZ, fontsize=8, transform=ax.transAxes)
    ax.set_title("curved group, flat tangent space\n" r"($S^2$ drawn as a stand-in for $SO(3)$)", fontsize=8.5)
    fig.savefig(OUT + "manifold.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F8 gyro integration
def f_gyro():
    dt, Tend = 0.01, 120.0
    w = lambda t: np.array([0.6 * np.sin(0.7 * t), 0.4 * np.cos(1.1 * t), 1.2 + 0.3 * np.sin(0.3 * t)])
    N = int(Tend / dt)
    Rn, Rr, Re, Rt = np.eye(3), np.eye(3), np.eye(3), np.eye(3)
    ts, on, orr, oe, an, ar, ae, gn, ge = [], [], [], [], [], [], [], [], []
    sub = 20
    for k in range(N):
        t = k * dt
        wk = w(t)
        Rn = Rn + Rn @ skew(wk) * dt
        Rr = Rr + Rr @ skew(wk) * dt
        U, S, Vt = np.linalg.svd(Rr); Rr = U @ Vt
        Re = Re @ Exp(wk * dt)
        for j in range(sub):
            Rt = Rt @ Exp(w(t + (j + 0.5) * dt / sub) * dt / sub)
        if k % 20 == 0:
            ts.append(t + dt)
            on.append(np.linalg.norm(Rn.T @ Rn - np.eye(3))); oe.append(np.linalg.norm(Re.T @ Re - np.eye(3)) + 1e-17)
            orr.append(np.linalg.norm(Rr.T @ Rr - np.eye(3)) + 1e-17)
            Rnp = np.linalg.svd(Rn); Rnp = Rnp[0] @ Rnp[2]
            an.append(np.linalg.norm(Log(Rt.T @ Rnp)) / deg); ar.append(np.linalg.norm(Log(Rt.T @ Rr)) / deg)
            ae.append(np.linalg.norm(Log(Rt.T @ Re)) / deg)
            gv = np.array([0, 0, 9.81]); gn.append(abs(np.linalg.norm(Rn.T @ gv) - 9.81) + 1e-16); ge.append(abs(np.linalg.norm(Re.T @ gv) - 9.81) + 1e-16)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3))
    axs[0].semilogy(ts, on, color=CX, label=r"naive: $R\leftarrow R+R[\omega]_\times\Delta t$")
    axs[0].semilogy(ts, orr, color=CY, ls="--", label="naive + SVD re-projection")
    axs[0].semilogy(ts, oe, color=CZ, label=r"exponential: $R\leftarrow R\,\mathrm{Exp}(\omega\Delta t)$")
    axs[0].set_xlabel("time [s]"); axs[0].set_ylabel(r"$\|R^{\top}R-I\|_F$")
    axs[0].set_title("(a) does the estimate stay a rotation?", fontsize=8.5)
    axs[0].legend(frameon=False, fontsize=7, loc="center right"); axs[0].grid(alpha=0.3, lw=0.4)
    axs[1].semilogy(ts, gn, color=CX, label="naive")
    axs[1].semilogy(ts, ge, color=CZ, label="exponential")
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel(r"$|\,\|R^{\top}g\|-9.81\,|$  [m/s$^2$]")
    axs[1].set_title(r"(b) consequence: gravity rotated into body frame", fontsize=8.5)
    axs[1].legend(frameon=False, fontsize=7); axs[1].grid(alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig(OUT + "gyro.pdf"); plt.close(fig)
    print("gyro final orth naive", on[-1], "exp", oe[-1], "att err naive", an[-1], "svd", ar[-1], "exp", ae[-1], "g err", gn[-1], ge[-1], "g at 10s", gn[5*1])


# ---------------------------------------------------------------- F9 double cover
def f_quat_cover():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    th = np.linspace(0, 4 * np.pi, 600)
    ax = axs[0]
    ax.plot(th / np.pi, np.cos(th / 2), color=CZ, lw=1.7, label=r"$q_w=\cos(\theta/2)$")
    ax.plot(th / np.pi, np.sin(th / 2), color=CY, lw=1.7, label=r"$q_z=\sin(\theta/2)$")
    ax.plot(th / np.pi, np.cos(th), color=CX, lw=1.2, ls="--", label=r"$r_{11}=\cos\theta$ (matrix)")
    for x in (2, 4):
        ax.axvline(x, color=GRY, lw=0.6)
    ax.annotate(r"$\theta=2\pi$: $R=I$ but $q=-1$", xy=(2, -1), xytext=(2.25, -0.7), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel(r"rotation angle $\theta$ about $z$  [units of $\pi$]"); ax.set_xlim(0, 4)
    ax.set_title(r"(a) $R$ has period $2\pi$, $q$ has period $4\pi$", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3); ax.grid(alpha=0.3, lw=0.4)
    ax = axs[1]; ax.set_aspect("equal"); ax.axis("off")
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=GRY, lw=0.8)
    tt = np.linspace(-np.pi / 2, np.pi / 2, 100); ax.plot(np.cos(tt), np.sin(tt), color=CZ, lw=3, alpha=0.25)
    a = 50 * deg
    q = np.array([np.cos(a), np.sin(a)])
    ax.annotate("", xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORA, lw=1.8))
    ax.annotate("", xy=-q, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=PUR, lw=1.8))
    ax.text(q[0] + 0.05, q[1] + 0.03, r"$q$: angle $\theta$", color=ORA, fontsize=8)
    ax.text(-q[0] - 0.95, -q[1] - 0.15, r"$-q$: angle $\theta+2\pi$", color=PUR, fontsize=8)
    ax.text(0.12, 0.08, r"$\theta/2$", fontsize=8)
    aa = np.linspace(0, a, 20); ax.plot(0.25 * np.cos(aa), 0.25 * np.sin(aa), color=DK, lw=0.7)
    ax.text(1.06, -0.05, r"$q_w$", fontsize=8); ax.text(-0.08, 1.07, r"$q_z$", fontsize=8)
    ax.plot([-1.15, 1.15], [0, 0], color=GRY, lw=0.5); ax.plot([0, 0], [-1.15, 1.15], color=GRY, lw=0.5)
    ax.text(-1.25, -1.45, r"both $\mapsto$ same $R_z(\theta)$; shaded: $q_w\geq0$ convention", fontsize=7.5)
    ax.set_title(r"(b) antipodal points, one rotation", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT + "quat_cover.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F10 slerp vs euler lerp
def f_slerp():
    e0 = np.zeros(3); e1 = np.array([150, 50, -120]) * deg  # (psi, theta, phi)
    q0 = R2q(zyx(*e0)); q1 = R2q(zyx(*e1))
    ts = np.linspace(0, 1, 201)
    paths = {}
    Rs_e = [zyx(*((1 - t) * e0 + t * e1)) for t in ts]
    Rs_s = [q2R(slerp(q0, q1, t)) for t in ts]
    def nlerp(t):
        q = (1 - t) * q0 + t * (q1 if q0 @ q1 >= 0 else -q1); return q / np.linalg.norm(q)
    Rs_n = [q2R(nlerp(t)) for t in ts]
    fig = plt.figure(figsize=(6.6, 2.7))
    ax = ax3d(fig, (1, 2, 1), lim=1.0, elev=20, azim=10, zoom=1.5)
    u, v = np.mgrid[0:2 * np.pi:36j, 0:np.pi:18j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GRY, lw=0.25, alpha=0.4)
    lens = {}
    for Rs, c, nm, ls in [(Rs_e, CX, "Euler-angle lerp", "-"), (Rs_s, CZ, "SLERP", "-"), (Rs_n, CY, "NLERP", ":")]:
        P = np.array([R[:, 2] for R in Rs]); ax.plot(P[:, 0], P[:, 1], P[:, 2], color=c, lw=2.0, ls=ls, label=nm)
        lens[nm] = sum(np.linalg.norm(Log(Rs[i].T @ Rs[i + 1])) for i in range(len(Rs) - 1)) / deg
    ax.scatter(*Rs_s[0][:, 2], color=DK, s=15); ax.scatter(*Rs_s[-1][:, 2], color=DK, s=15)
    ax.text(*(Rs_s[0][:, 2] * 1.12), "start", fontsize=8); ax.text(*(Rs_s[-1][:, 2] * 1.15), "end", fontsize=8)
    ax.set_title(r"(a) path traced by the body $z$-axis tip", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(-0.05, -0.02))
    ax2 = fig.add_subplot(1, 2, 2)
    for Rs, c, nm, ls in [(Rs_e, CX, "Euler-angle lerp", "-"), (Rs_s, CZ, "SLERP", "-"), (Rs_n, CY, "NLERP", "--")]:
        sp = [np.linalg.norm(Log(Rs[i].T @ Rs[i + 1])) / (ts[1] - ts[0]) / deg for i in range(len(Rs) - 1)]
        ax2.plot(ts[:-1], sp, color=c, ls=ls, lw=1.6, label=f"{nm}: total {lens[nm]:.0f}°")
    ax2.set_xlabel(r"interpolation parameter $t$"); ax2.set_ylabel(r"angular speed [deg / unit $t$]")
    ax2.set_title("(b) speed along the path (geodesic length = %.0f°)" % (2 * np.arccos(abs(q0 @ q1)) / deg), fontsize=8.5)
    ax2.legend(frameon=False, fontsize=7.5); ax2.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig(OUT + "slerp.pdf"); plt.close(fig)
    print("slerp lens", lens)


# ---------------------------------------------------------------- F11 transform chain
def f_chain():
    R_BC0 = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    T_WB = T(Rz(30 * deg), [2.0, 1.0, 0.0]); T_BC = T(Ry(20 * deg) @ R_BC0, [0.25, 0, 0.6])
    T_WC = T_WB @ T_BC
    pW = T_WC @ np.array([0.3, -0.1, 1.5, 1])
    fig = plt.figure(figsize=(5.2, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.3, 4.1); ax.set_ylim(-0.4, 2.4); ax.set_zlim(0, 1.2)
    ax.set_box_aspect((4.2, 2.6, 1.2), zoom=1.4); ax.view_init(elev=26, azim=-58); ax.set_axis_off()
    gx, gy = np.meshgrid(np.linspace(-0.2, 4.0, 15), np.linspace(-0.3, 2.3, 10))
    ax.plot_wireframe(gx, gy, 0 * gx, color=GRY, lw=0.25, alpha=0.5)
    frame(ax, np.eye(3), o=(0, 0, 0), s=0.55, labels=(r"$x_W$", r"$y_W$", r"$z_W$"), fs=8)
    frame(ax, T_WB[:3, :3], o=T_WB[:3, 3], s=0.5, labels=(r"$x_B$", r"$y_B$", r"$z_B$"), fs=8)
    frame(ax, T_WC[:3, :3], o=T_WC[:3, 3], s=0.35, labels=(r"$x_C$", r"$y_C$", r"$z_C$"), fs=8)
    # robot body box
    c = T_WB[:3, 3] + T_WB[:3, :3] @ np.array([0.0, 0, 0.15])
    ax.scatter(*pW[:3], color=ORA, s=30)
    ax.text(*(pW[:3] + np.array([0.05, 0.05, 0.12])), r"object $p$", color=ORA, fontsize=8.5)
    ax.plot(*np.c_[T_WC[:3, 3], pW[:3]], color=ORA, ls=":", lw=1.2)
    ax.plot(*np.c_[np.zeros(3), T_WB[:3, 3]], color=DK, ls="--", lw=0.8)
    ax.text(*(T_WB[:3, 3] / 2 + np.array([0, -0.3, 0])), r"$T_{WB}$", fontsize=9)
    ax.plot(*np.c_[T_WB[:3, 3], T_WC[:3, 3]], color=DK, ls="--", lw=0.8)
    ax.text(*((T_WB[:3, 3] + T_WC[:3, 3]) / 2 + np.array([-0.35, 0, 0.1])), r"$T_{BC}$", fontsize=9)
    ax.set_title(r"World $\to$ base $\to$ camera $\to$ object:  ${}^Wp=T_{WB}\,T_{BC}\,{}^Cp$", fontsize=9)
    fig.savefig(OUT + "chain.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F12 screw
def f_screw():
    T2 = T(Rz(np.pi / 2), [1, 1, 0.5]); xi = LogSE3(T2)
    tip = np.array([0.55, 0, 0, 1.0])
    ts = np.linspace(0, 1, 101); tk = np.linspace(0, 1, 5)
    q0 = np.array([1, 0, 0, 0.]); q1 = R2q(T2[:3, :3])
    def screw(t): return ExpSE3(t * xi)
    def decoupled(t): return T(q2R(slerp(q0, q1, t)), t * T2[:3, 3])
    fig = plt.figure(figsize=(6.6, 2.9))
    for k, (f, title) in enumerate([(screw, r"(a) screw: $T(t)=\exp(t\,[\xi])$, constant twist"),
                                     (decoupled, "(b) decoupled: lerp translation, slerp rotation")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_xlim(-0.3, 1.5); ax.set_ylim(-0.3, 1.6); ax.set_zlim(-0.1, 0.9)
        ax.set_box_aspect((1.8, 1.9, 1.0), zoom=1.45); ax.view_init(elev=38, azim=-62); ax.set_axis_off()
        gx, gy = np.meshgrid(np.linspace(-0.3, 1.5, 7), np.linspace(-0.3, 1.6, 7))
        ax.plot_wireframe(gx, gy, 0 * gx, color=GRY, lw=0.25, alpha=0.4)
        O = np.array([f(t)[:3, 3] for t in ts]); P = np.array([(f(t) @ tip)[:3] for t in ts])
        ax.plot(*O.T, color=PUR, lw=1.8, label="origin path")
        ax.plot(*P.T, color=ORA, lw=1.8, label="gripper-tip path")
        for t in tk:
            M = f(t); frame(ax, M[:3, :3], o=M[:3, 3], s=0.35, labels=None, lw=1.2)
            tp = (M @ tip)[:3]; ax.plot(*np.c_[M[:3, 3], tp], color=GRY, lw=1.0)
        if k == 0:
            ax.plot([0, 0], [1, 1], [-0.1, 0.9], color=DK, lw=1.4, ls="--")
            ax.text(0, 1, 0.97, r"screw axis $x{=}0,y{=}1$; $h=0.318$ m/rad", fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fig.subplots_adjust(wspace=0.02)
    fig.savefig(OUT + "screw.pdf"); plt.close(fig)


# ---------------------------------------------------------------- F13 FK arm
def f_arm():
    L0, L1, L2 = 0.3, 0.4, 0.3
    def S(w, q):
        w = np.array(w, float); q = np.array(q, float); return np.concatenate([-np.cross(w, q), w])
    Ss = [S([0, 0, 1], [0, 0, 0]), S([0, -1, 0], [0, 0, L0]), S([0, -1, 0], [L1, 0, L0])]
    M = T(np.eye(3), [L1 + L2, 0, L0])
    th = np.array([30, 45, -60]) * deg
    E = [expm(hat6(Ss[i]) * th[i]) for i in range(3)]
    J2 = E[0] @ np.array([0, 0, L0, 1]); J3 = E[0] @ E[1] @ np.array([L1, 0, L0, 1]); Tee = E[0] @ E[1] @ E[2] @ M
    pts = np.array([[0, 0, 0], J2[:3], J3[:3], Tee[:3, 3]])
    home = np.array([[0, 0, 0], [0, 0, L0], [L1, 0, L0], [L1 + L2, 0, L0]])
    fig = plt.figure(figsize=(3.4, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.1, 0.75); ax.set_ylim(-0.15, 0.6); ax.set_zlim(0, 0.7); ax.set_box_aspect((0.85, 0.75, 0.7), zoom=1.1)
    ax.view_init(elev=22, azim=-55); ax.set_axis_off()
    ax.plot(*home.T, color=GRY, lw=3, alpha=0.5, solid_capstyle="round", label=r"home $M$ ($\theta=0$)")
    ax.plot(*pts.T, color=CZ, lw=4, solid_capstyle="round", label=r"$\theta=(30^\circ,45^\circ,-60^\circ)$")
    ax.scatter(*pts[:3].T, color=DK, s=22, zorder=5)
    frame(ax, np.eye(3), s=0.18, labels=("x", "y", "z"), lw=1.0, fs=7)
    frame(ax, Tee[:3, :3], o=Tee[:3, 3], s=0.15, labels=None, lw=1.4)
    a2 = E[0][:3, :3] @ np.array([0, -1, 0]); a3 = a2
    for p, a in [(pts[0], np.array([0, 0, 1.])), (pts[1], a2), (pts[2], a3)]:
        ax.plot(*np.c_[p - 0.1 * a, p + 0.1 * a], color=PUR, lw=1.6)
    ax.text(*(pts[3] + np.array([-0.25, 0, 0.09])), r"$p_{ee}=(0.496,\,0.286,\,0.505)$", fontsize=7.5)
    ax.set_title("3-DOF elbow arm by product of exponentials\n(purple: joint screw axes)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(OUT + "arm.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in [f_frames, f_active_passive, f_noncommute, f_zyx_seq, f_gimbal, f_rodrigues, f_manifold, f_gyro,
              f_quat_cover, f_slerp, f_chain, f_screw, f_arm]:
        f(); print("ok", f.__name__)
