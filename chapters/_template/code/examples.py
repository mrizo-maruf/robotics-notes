"""Every number quoted in this chapter's worked examples is computed here."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code"))  # repo-level shared library
import numpy as np
from rigid import *

np.set_printoptions(precision=4, suppress=True)

print("=== Example: ... ===")
