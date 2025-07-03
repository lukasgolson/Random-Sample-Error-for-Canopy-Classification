import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Functions to create figure
def create_pattern_matrix(pattern_type, size):
    """Generate NxN matrices for different spatial patterns."""
    if pattern_type == "negative":
        return np.indices((size, size)).sum(axis=0) % 2
    elif pattern_type == "weak_negative":
        mat = np.indices((size, size)).sum(axis=0) % 3
        return (mat == 1).astype(int)  # breaks perfect alternation
    elif pattern_type == "random":
        np.random.seed(42)
        return np.random.randint(0, 2, (size, size))
    elif pattern_type == "weak_positive":
        mat = np.zeros((size, size), dtype=int)
        step = size // 3
        mat[step:2 * step, step:] = 1  # Middle band
        mat[2 * step:, :step] = 1  # Bottom-left block green
        mat[2 * step:, 2 * step:] = 1  # Bottom-right block green
        return mat
    elif pattern_type == "positive":
        mat = np.zeros((size, size), dtype=int)
        half = size // 2
        mat[half:, :] = 1  # Bottom half green (1), top half remains white (0)
        return mat

# Define custom colors: [0] = white, [1] = light forest green
custom_colors = ListedColormap(['white', '#42712f'])

fig, axes = plt.subplots(1, 5, figsize=(10, 3.8))

patterns = ['negative', 'weak_negative', 'random', 'weak_positive', 'positive']
labels = ["Strong Negative\n(Moran's i = -1)", "", "No Autocorrelation\n(Moran's i = 0)", "",
          "Strong Positive\n(Moran's i = 1)"]

for ax, pattern, label in zip(axes, patterns, labels):
    data = create_pattern_matrix(pattern, size=15)
    ax.imshow(data, cmap=custom_colors, interpolation='nearest')

    # Axis border only (no ticks or labels)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Adjust spacing to make room below
plt.tight_layout(rect=[0, 0.2, 1, 0.95])

# Draw an arrow
ax = axes[2]
fig_line_y = 0.22
ax.annotate(
    '', xy=(0.95, fig_line_y), xytext=(0.05, fig_line_y),
    xycoords='figure fraction', textcoords='figure fraction',
    arrowprops=dict(
        arrowstyle='<|-|>',
        color='black',
        linewidth=2.5,  # thicker line
        shrinkA=0, shrinkB=0,  # no shrinking of line length
        mutation_scale=20  # larger arrowheads
    )
)

# Get the center x-positions of axes 0, 2, and 4
x_pos_1 = axes[0].get_position().x0 + axes[0].get_position().width / 2
x_pos_3 = axes[2].get_position().x0 + axes[2].get_position().width / 2
x_pos_5 = axes[4].get_position().x0 + axes[4].get_position().width / 2

# Add bottom explanatory text centered under subplots 1, 3, and 5
fig.text(x_pos_1, 0.11, "Dispersed Distribution\n(Moran's i = -1)", ha='center', va='center', fontsize=11)
fig.text(x_pos_3, 0.11, "Random Distribution\n(Moran's i = 0)", ha='center', va='center', fontsize=11)
fig.text(x_pos_5, 0.11, "Clustered Distribution\n(Moran's i = 1)", ha='center', va='center', fontsize=11)

fig.savefig("Figue 1.png", dpi=600, bbox_inches='tight')
plt.show()
