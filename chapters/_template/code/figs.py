"""Figures for this chapter -> ../figs/*.pdf. Style: see CLAUDE.md (Figures)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rigid import *
from style import *   # shared colours + rcParams

OUT = os.path.join(HERE, "..", "figs") + os.sep

if __name__ == "__main__":
    pass
