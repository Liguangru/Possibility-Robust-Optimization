# -*- coding: utf-8 -*-
"""Verify that the openTPS installation is working correctly."""
import numpy
import scipy
import opentps.core as oc

print("openTPS imported successfully")
print("Core module path:", oc.__file__)
print("numpy:", numpy.__version__, "| scipy:", scipy.__version__)
