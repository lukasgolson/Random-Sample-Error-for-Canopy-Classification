import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the csv
df = pd.read_csv('sample_points.csv')

def_calculations
  # Error calculations
  se_standard = np.sqrt(p * (1 - p) / sample_points) # Standard (i-Tree) SE calculation
  se_neff = np.sqrt(p * (1 - p) / neff) # N effective (Moran's i) SE calculation
  
  # Margin calculations
  margin_standard = 1.96 * se_standard # Margin of error derived from standard SE calculation
  margin_neff = 1.96 * se_neff # Margin of error derived from N effective SE calculation
  
  # LB and UB standards
  lb_standard = max(0, p - margin_standard)
  ub_standard = min(1, p + margin_standard)
  lb_neff = max(0, p - margin_neff)
  ub_neff = min(1, p + margin_neff)

  return(se_standard, se_neff, margin_standard, margin_neff, lb_standard, ub_standard, lb_neff, ub_neff)
