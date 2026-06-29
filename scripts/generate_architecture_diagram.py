import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import os

# Use serif font like a standard IEEE research paper
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm' # Computer modern for math

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.02)
ax.axis('off')

# Main enclosing box
main_rect = patches.Rectangle((0.05, 0.0), 0.9, 1.0, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(main_rect)

# Title
ax.text(0.5, 0.93, "Dual-Branch Architecture", ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(0.5, 0.85, "INPUT: Connectome $G$ + Temporal Features", ha='center', va='center', fontsize=10)

# Downward arrows from Input
ax.annotate('', xy=(0.25, 0.71), xytext=(0.25, 0.80), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.75, 0.71), xytext=(0.75, 0.80), arrowprops=dict(arrowstyle="->", lw=1))

# Text for GATv2
ax.text(0.25, 0.65, "GATv2 $\\times$ 2 layers", ha='center', va='center', fontsize=10)
ax.text(0.25, 0.58, "(4 heads, LayerNorm)", ha='center', va='center', fontsize=10)

# Text for GRU
ax.text(0.75, 0.65, "GRU", ha='center', va='center', fontsize=10)
ax.text(0.75, 0.58, "(hidden=96)", ha='center', va='center', fontsize=10)

# Arrows to z
ax.annotate('', xy=(0.25, 0.45), xytext=(0.25, 0.53), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.75, 0.45), xytext=(0.75, 0.53), arrowprops=dict(arrowstyle="->", lw=1))

# Embeddings text
ax.text(0.25, 0.39, "$z^{(S)} \in \mathbb{R}^{96}$", ha='center', va='center', fontsize=11)
ax.text(0.75, 0.39, "$z^{(T)} \in \mathbb{R}^{96}$", ha='center', va='center', fontsize=11)

# Diagonal arrows converging to concat
ax.annotate('', xy=(0.42, 0.26), xytext=(0.28, 0.34), arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate('', xy=(0.58, 0.26), xytext=(0.72, 0.34), arrowprops=dict(arrowstyle="->", lw=1))

# Box around Concat -> MLP -> Sigmoid
rect2 = patches.Rectangle((0.28, 0.16), 0.44, 0.09, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(rect2)
ax.text(0.5, 0.205, "Concat $\\rightarrow$ MLP $\\rightarrow$ Sigmoid", ha='center', va='center', fontsize=10)

# Downward arrow to final output
ax.annotate('', xy=(0.5, 0.08), xytext=(0.5, 0.16), arrowprops=dict(arrowstyle="->", lw=1))

# Final output text
ax.text(0.5, 0.04, "$\hat{y}_i \in [0, 1]$", ha='center', va='center', fontsize=11)

# Create output dir if needed
out_dir = "e:/Zombozomes - research/paper"
os.makedirs(out_dir, exist_ok=True)

# Save as high-res PNG and vector PDF
png_path = os.path.join(out_dir, "Dual_Branch_Architecture.png")
pdf_path = os.path.join(out_dir, "Dual_Branch_Architecture.pdf")

plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0.02)
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
print(f"Generated diagram successfully at {png_path} and {pdf_path}")
