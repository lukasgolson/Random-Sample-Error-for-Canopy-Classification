## --------------------------------------------- i-Tree Canopy Calculation of Standard Error and Confidence Intervals ---------------------------------------------
#region

# Input values
N = 1000  # total sampled points
n = 330   # points classified as tree

# Compute p and q
p = n / N
q = 1 - p

# Conditional SE formula
if n < 10:
    SE = (n ** 0.5) / N
else:
    SE = (p * q / N) ** 0.5

# Confidence interval calculation (only if N >= 30)
if N >= 30:
    margin = 1.96 * SE
    lower_bound = max(0, p - margin)
    upper_bound = min(1, p + margin)
    print(f"95% Confidence Interval: {lower_bound:.1%} to {upper_bound:.1%}")

# Output
print(f"Percent tree cover: {p:.2%}")
print(f"Standard Error (SE): {SE:.4f}")

#endregion
