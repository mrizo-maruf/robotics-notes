"""Shared matplotlib style for every chapter's figures.

Usage in chapters/<ch>/code/figs.py:   from style import *
Full-width figures are 6.6 in wide (the A4 text width at 2 cm margins);
half-width 3.3 in. Save as PDF (vector) into the chapter's figs/ folder.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# Axis colours: x red, y green, z blue (same as RViz). Accents for everything else.
CX, CY, CZ = "#c0392b", "#27884a", "#2c6fbb"
ORA, PUR, GRY, DK = "#e67e22", "#8e44ad", "#9a9a9a", "#333333"
FULL_W, HALF_W = 6.6, 3.3
