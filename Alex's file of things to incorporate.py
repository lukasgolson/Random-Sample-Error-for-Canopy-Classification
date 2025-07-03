## --------------------------------------------- i-Tree Canopy Calculation of Standard Error ---------------------------------------------
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

# Output
print(f"Percent tree cover: {p:.2%}")
print(f"Standard Error (SE): {SE:.4f}")

#endregion
