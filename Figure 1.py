import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Functions to create figure
def create_pattern_matrix(pattern_type, size):
    """Generate NxN matrices for different spatial patterns."""
    if pattern_type == "negative":
        return np.indices((size, size)).sum(axis=0) % 2
    elif pattern_type == "random":
        np.random.seed(42)
        return np.random.randint(0, 2, (size, size))
    elif pattern_type == "positive":
        mat = np.zeros((size, size), dtype=int)
        half = size // 2
        mat[:half, :half] = 1
        mat[half:, half:] = 1
        return mat
    else:
        raise ValueError("Invalid pattern type")

def plot_moran_patterns(size=15):
    # Define custom colors: [0] = white, [1] = light forest green
    custom_colors = ListedColormap(['white', '#42712f'])  # #228B22 is forest green

    fig, axes = plt.subplots(1, 3, figsize=(3 * 2, 3))
    patterns = ['negative', 'random', 'positive']
    labels = ["Negative Autocorrelation\n(Moran's i = -1)",
              "No Autocorrelation\n(Moran's i = 0)",
              "Positive Autocorrelation\n(Moran's i = 1)"]

    for ax, pattern, label in zip(axes, patterns, labels):
        data = create_pattern_matrix(pattern, size)
        ax.imshow(data, cmap=custom_colors, interpolation='nearest')
        ax.axis('off')
        ax.text(0.5, -0.15, label, fontsize=10, ha='center', va='top', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

# Usage
plot_moran_patterns()
