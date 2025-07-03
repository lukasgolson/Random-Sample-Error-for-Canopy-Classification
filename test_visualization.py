import matplotlib.pyplot as plt
from canopy_model import generate_canopy_map


canopy_fraction = 0.5  # Fraction of the area that should be covered by canopies
model_seed = None

low_cluster_map = generate_canopy_map(clustering=5, canopy_cover=canopy_fraction, seed=model_seed)
medium_cluster_map = generate_canopy_map(clustering=25, canopy_cover=canopy_fraction, seed=model_seed)
high_cluster_map = generate_canopy_map(clustering=50, canopy_cover=canopy_fraction, seed=model_seed)


# --- Plotting Logic ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Simulated Canopy Maps with Varying Clustering', fontsize=16)

# Colormap for visualization
cmap = 'Greens'

axes[0].imshow(low_cluster_map, cmap=cmap, interpolation='nearest')
axes[0].set_title('Clustering: 5')

axes[1].imshow(medium_cluster_map, cmap=cmap, interpolation='nearest')
axes[1].set_title('Clustering: 25')

axes[2].imshow(high_cluster_map, cmap=cmap, interpolation='nearest')
axes[2].set_title('Clustering: 50')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()