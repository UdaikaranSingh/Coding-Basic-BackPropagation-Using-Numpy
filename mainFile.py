import neuralnet
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

"""
importing all the problems
"""
import gradientchecker
import ProblemC
import ProblemD
import ProblemE
import ProblemF

"""
This is the main file to run all the experiments

Class: CSE 190
Assignment: PA2
"""

print("Part B: Gradient Checker")
print()
gradientchecker.main()
print()

print("Part C: Finding Optimal # of Epochs")
print()
ProblemC.main()
print()

print("Part D: Experiment with Regularization")
print()
ProblemD.main()
print()

print("Part E: Experiment with Activations")
print()
ProblemE.main()
print()

print("Part F: Experiment with Network Topology")
print()
ProblemF.main()
print()

