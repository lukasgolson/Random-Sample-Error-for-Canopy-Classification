import math

# Input values
N = 1000  # Total sampled points
n = 330   # Points classified as tree
nc = 5       # Average neighborhood (cluster) size
rho = 0.9    # Spatial autocorrelation (e.g., Moran’s I)

# Compute proportion and complement
p = n / N
q = 1 - p

# Compute standard error (SE)
if n < 10:
    # For very small n, use simplified SE estimate (field-specific adjustment)
    SE = math.sqrt(n) / N
else:
    SE = math.sqrt(p * q / N)

# Confidence interval calculation (if sample size is sufficient)
if N >= 30:
    margin = 1.96 * SE  # 95% confidence level
    lower_bound = max(0, p - margin)
    upper_bound = min(1, p + margin)
    print(f"95% Confidence Interval: {lower_bound:.1%} to {upper_bound:.1%}")

# Output basic results
print(f"Percent tree cover: {p:.1%}")
print(f"Standard Error (SE): {SE:.4f}")

# Functions for spatially adjusted SEs
def se_adjusted_deff(p, N, nc, rho):
    deff = 1 + (nc - 1) * rho
    se = math.sqrt(p * (1 - p) / N)
    return se * math.sqrt(deff)

def se_adjusted_neff(p, N, rho):
    neff = N / (1 + (N - 1) * rho)
    return math.sqrt(p * (1 - p) / neff)

# Output adjusted SEs
print(f"Standard Error with Design Effect: {se_adjusted_deff(p, N, nc, rho):.4f}")
print(f"Standard Error with Effective N:   {se_adjusted_neff(p, N, rho):.4f}")
