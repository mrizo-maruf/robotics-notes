"""Figures for chapter 3 (Lie groups) -> ../figs/*.pdf.   Run: make figs CH=03-lie-groups"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from rigid import *
from style import *

OUT = os.path.join(HERE, "..", "figs") + os.sep
TEAL = "#0b7f6d"


def robot(ax, M, col=TEAL, s=0.16, alpha=1.0, fill="#e8f4f1", z=5):
    th = np.arctan2(M[1, 0], M[0, 0]); x, y = M[0, 2], M[1, 2]
    tr = Affine2D().rotate(th).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-0.6 * s, -0.45 * s), 1.2 * s, 0.9 * s, fc=fill, ec=col, lw=1.0, transform=tr, alpha=alpha, zorder=z))
    ax.annotate("", xy=(x + 0.95 * s * np.cos(th), y + 0.95 * s * np.sin(th)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=8), zorder=z + 1)


def clean(ax, lim=None):
    ax.set_aspect("equal")
    if lim: ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)


# --------------------------------------------------------------------------- F1 the circle group
def f_circle():
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
    t = np.linspace(0, 2 * np.pi, 300)
    ax = axs[0]
    ax.plot(np.cos(t), np.sin(t), color=CZ, lw=2, label="group $S^1$: $|z|=1$")
    ax.plot([1, 1], [-1.9, 2.7], color=ORA, lw=1.4, label="tangent space at $1$: $i\\mathbb{R}$")
    for tau, c in [(0.6, CY), (1.4, PUR), (2.6, CX)]:
        ax.plot(1, tau, "o", color=c, ms=4)
        a = np.linspace(0, tau, 60); ax.plot(np.cos(a), np.sin(a), color=c, lw=2.4, alpha=0.6)
        ax.plot(np.cos(tau), np.sin(tau), "o", color=c, ms=4)
        ax.plot([1, 1], [0, tau], color=c, lw=2.4, alpha=0.6)
        ax.text(1.08, tau, f"$i\\tau={tau}i$", color=c, fontsize=7.5, va="center")
        ax.text(1.18 * np.cos(tau) - 0.05, 1.18 * np.sin(tau), "$e^{i%.1f}$" % tau, color=c, fontsize=7.5, ha="center", va="center")
    ax.plot(1, 0, "ko", ms=4); ax.text(1.06, -0.18, "$\\mathcal{E}=1$", fontsize=8)
    ax.text(-1.7, -1.75, "exp wraps the line onto the circle,\npreserving length (arc = $\\tau$)", fontsize=7.5)
    clean(ax, (-1.8, 2.0, -1.9, 2.7)); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("(a) the Lie algebra wraps onto the group", fontsize=8.5)
    ax = axs[1]
    ax.plot(np.cos(t), np.sin(t), color=CZ, lw=2)
    al = 0.9; X = np.exp(1j * al); d = 1j * X
    ax.plot([X.real - 1.2 * d.real, X.real + 1.6 * d.real], [X.imag - 1.2 * d.imag, X.imag + 1.6 * d.imag], color=ORA, lw=1.4)
    tau = 0.9; Y = X * np.exp(1j * tau); a = np.linspace(al, al + tau, 50)
    ax.plot(np.cos(a), np.sin(a), color=CY, lw=2.6, alpha=0.7)
    ax.annotate("", xy=(X.real + tau * d.real, X.imag + tau * d.imag), xytext=(X.real, X.imag), arrowprops=dict(arrowstyle="-|>", color=CY, lw=1.5))
    ax.plot(X.real, X.imag, "ko", ms=4); ax.text(X.real + 0.08, X.imag - 0.12, "$\\mathcal{X}$", fontsize=9)
    ax.plot(Y.real, Y.imag, "o", color=CY, ms=4); ax.text(Y.real - 0.55, Y.imag + 0.05, "$\\mathcal{X}\\oplus\\tau$", color=CY, fontsize=9)
    ax.text(X.real + 0.55 * d.real + 0.08, X.imag + 0.55 * d.imag + 0.12, "$\\tau$", color=CY, fontsize=9)
    z1, z2 = np.exp(0.5j), np.exp(1.7j); s = z1 + z2
    for z in (z1, z2): ax.plot(z.real, z.imag, "s", color=GRY, ms=4)
    ax.plot(s.real, s.imag, "x", color=CX, ms=7, mew=2); ax.text(s.real + 0.08, s.imag, "$z_1+z_2\\notin S^1$", color=CX, fontsize=8)
    ax.text(-1.75, -1.75, "tangent line at $\\mathcal{X}$: local increments $\\tau$;\nplain addition leaves the group", fontsize=7.5)
    clean(ax, (-1.8, 2.0, -1.9, 2.7)); ax.set_title("(b) increments live in the tangent space at $\\mathcal{X}$", fontsize=8.5)
    fig.tight_layout(w_pad=0.5); fig.savefig(OUT + "circle_group.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F2 right vs left plus
def f_plus():
    fig, ax = plt.subplots(figsize=(HALF_W + 0.6, 2.8))
    X = T2(np.radians(30), [2.0, 1.0]); tau = np.array([1.0, 0, 0.5])
    s = np.linspace(0, 1, 60)
    Pr = np.array([(X @ ExpSE2(l * tau))[:2, 2] for l in s]); Pl = np.array([(ExpSE2(l * tau) @ X)[:2, 2] for l in s])
    ax.plot(Pr[:, 0], Pr[:, 1], color=CZ, lw=1.8, label="$\\mathcal{X}\\oplus\\tau=\\mathcal{X}\\,\\mathrm{Exp}(\\tau)$ (local)")
    ax.plot(Pl[:, 0], Pl[:, 1], color=ORA, lw=1.8, ls="--", label="$\\tau\\oplus\\mathcal{X}=\\mathrm{Exp}(\\tau)\\,\\mathcal{X}$ (global)")
    C = J2 @ tau[:2] / tau[2]; ax.plot(*C, "o", color=ORA, ms=3); ax.text(C[0] + 0.07, C[1] - 0.05, "global turn centre $(0,2)$", color=ORA, fontsize=7)
    Cl = X[:2, 2] + X[:2, :2] @ C; ax.plot(*Cl, "o", color=CZ, ms=3); ax.text(Cl[0] + 0.07, Cl[1] + 0.02, "local turn centre", color=CZ, fontsize=7)
    robot(ax, X, col=GRY, fill="#f2f2f2"); robot(ax, X @ ExpSE2(tau), col=CZ, fill="#e3edf7"); robot(ax, ExpSE2(tau) @ X, col=ORA, fill="#fbeedd")
    for i, c in enumerate((CX, CY)):
        e = np.eye(2)[i] * 0.5; ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))
    ax.text(-0.05, -0.22, "world origin", fontsize=7)
    clean(ax, (-0.4, 3.4, -0.4, 3.6)); ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.text(2.1, 0.75, "$\\mathcal{X}$", fontsize=9, color=DK)
    fig.tight_layout(); fig.savefig(OUT + "plus_left_right.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F3 banana
def f_banana():
    import examples as ex
    Xs, Xbar, P, mE, PE = ex.banana(N=3000, sw=0.1)
    rng = np.random.default_rng(9); N = 3000
    L = np.array([(Xbar @ ExpSE2(t))[:2, 2] for t in rng.multivariate_normal(np.zeros(3), P, N)])
    E = rng.multivariate_normal(mE, PE, N)[:, :2]
    S = Xs[:, :2, 2]
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.2), sharey=True)
    for ax, D, c, ttl in [(axs[0], S, DK, "(a) Monte Carlo: true distribution"),
                          (axs[1], L, CZ, "(b) Gaussian in the tangent space"),
                          (axs[2], E, CX, "(c) Gaussian in $(x,y,\\theta)$")]:
        ax.scatter(D[:, 0], D[:, 1], s=0.6, color=c, alpha=0.35, lw=0)
        ax.plot(0, 0, "k^", ms=4); clean(ax, (-0.1, 2.4, -1.25, 1.25)); ax.set_title(ttl, fontsize=8.2)
        ax.set_xlabel("$x$ [m]", fontsize=8)
    axs[0].set_ylabel("$y$ [m]", fontsize=8)
    axs[1].text(0.0, -1.15, "73% of true samples inside its 95% region", fontsize=6.8, color=CZ)
    axs[2].text(0.0, -1.15, "48% of true samples inside its 95% region", fontsize=6.8, color=CX)
    fig.tight_layout(w_pad=0.3); fig.savefig(OUT + "banana.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- F4 ESKF + pose graph
def ellipse(ax, mu, S, c, k=3.0):
    w, V = np.linalg.eigh(S); t = np.linspace(0, 2 * np.pi, 60)
    pts = (V @ np.diag(k * np.sqrt(np.maximum(w, 0))) @ np.vstack([np.cos(t), np.sin(t)])).T + mu
    ax.plot(pts[:, 0], pts[:, 1], color=c, lw=0.9)


def f_robotics():
    d = np.load(os.path.join(HERE, "eskf.npz")); g = np.load(os.path.join(HERE, "posegraph.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]; B = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, -1.0]])
    ax.plot(d["true"][:, 0, 2], d["true"][:, 1, 2], color=GRY, lw=3, alpha=0.5, label="true")
    ax.plot(d["dr"][:, 0, 2], d["dr"][:, 1, 2], color=ORA, lw=1.0, label="dead reckoning")
    ax.plot(d["est"][:, 0, 2], d["est"][:, 1, 2], color=CZ, lw=1.0, label="ESKF on SE(2)")
    for k in range(0, len(d["est"]), 60):
        R = d["est"][k][:2, :2]; ellipse(ax, d["est"][k][:2, 2], R @ d["P"][k][:2, :2] @ R.T * 25, CZ, k=3.0)
    ax.plot(B[:, 0], B[:, 1], "k*", ms=7, label="beacons")
    clean(ax); ax.legend(frameon=False, fontsize=6.5, loc="center"); ax.set_title("(a) error-state Kalman filter on $SE(2)$\n(3$\\sigma$ position ellipses, $\\times5$)", fontsize=8.5)
    ax = axs[1]
    for key, c, lw, lab, ls in [("true", GRY, 3, "true", "-"), ("init", ORA, 1.0, "odometry (initial guess)", "--"), ("opt", CZ, 1.0, "after Gauss-Newton on $SE(2)$", "-")]:
        P = g[key][:, :2, 2]; ax.plot(P[:, 0], P[:, 1], ls, color=c, lw=lw, alpha=0.5 if key == "true" else 1, marker="o" if key != "true" else None, ms=2.5, label=lab)
    i = g["init"]; ax.plot([i[0, 0, 2], i[-1, 0, 2]], [i[0, 1, 2], i[-1, 1, 2]], ":", color=CX, lw=1.2, label="loop-closure edge")
    clean(ax); ax.legend(frameon=False, fontsize=6.5, loc="center"); ax.set_title("(b) pose-graph optimisation on $SE(2)$", fontsize=8.5)
    fig.tight_layout(w_pad=0.8); fig.savefig(OUT + "lie_robotics.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in [f_circle, f_plus, f_banana, f_robotics]:
        f(); print("ok", f.__name__)
