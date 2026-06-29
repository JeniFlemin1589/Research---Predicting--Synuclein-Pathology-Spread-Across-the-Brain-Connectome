import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import os

# Set serif font to match conventional IEEE research papers
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(7, 8.5))
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.02)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, linestyle='-'):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', linestyle=linestyle, facecolor='none')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1))

# Main enclosing box for the entire methodology
main_rect = patches.Rectangle((0.02, 0.05), 0.96, 0.93, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(main_rect)

# Add title
ax.text(0.5, 0.95, "RIP-GNN: Complete Experimental Methodology", ha='center', va='center', fontsize=12, fontweight='bold')

# --- 1. INPUT MODULE ---
ax.text(0.5, 0.89, "1. INPUT MODULE", ha='center', va='center', fontsize=11, fontweight='bold')
draw_box(ax, 0.1, 0.81, 0.35, 0.06, "$\\alpha$-Synuclein Pathology Data\n(Henderson et al.)")
draw_box(ax, 0.55, 0.81, 0.35, 0.06, "Structural Connectome $G$\n(116 ABA Regions)")

draw_arrow(ax, 0.275, 0.81, 0.275, 0.75)
draw_arrow(ax, 0.725, 0.81, 0.725, 0.75)

# --- 2 & 3. PREPROCESSING and GRAPH CONSTRUCTION ---
ax.text(0.275, 0.72, "2. DATA PREPROCESSING", ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(0.725, 0.72, "3. GRAPH CONSTRUCTION", ha='center', va='center', fontsize=11, fontweight='bold')

draw_box(ax, 0.1, 0.61, 0.35, 0.08, "Kinetic Traces $(T=1, 3, 6)$\n$\\downarrow$\nAggregation: Pathology $z$, $\\Delta$")
draw_box(ax, 0.55, 0.61, 0.35, 0.08, "Weighted Edges (Ipsi, Contra)\n$\\downarrow$\nPrior Node Vulnerability")

draw_arrow(ax, 0.275, 0.61, 0.275, 0.53)
draw_arrow(ax, 0.725, 0.61, 0.725, 0.53)

# --- 4. DUAL-BRANCH ARCHITECTURE ---
ax.text(0.5, 0.51, "4. DUAL-BRANCH ARCHITECTURE", ha='center', va='center', fontsize=11, fontweight='bold')

# Inner boundary box for architecture
arch_rect = patches.Rectangle((0.06, 0.36), 0.88, 0.13, linewidth=1, linestyle='--', edgecolor='black', facecolor='none')
ax.add_patch(arch_rect)

draw_box(ax, 0.1, 0.38, 0.35, 0.09, "Temporal Branch\n(GRU Sequence Modeling)\n$\\rightarrow z^{(T)} \\in \\mathbb{R}^{96}$")
draw_box(ax, 0.55, 0.38, 0.35, 0.09, "Spatial Branch\n(GATv2 Graph Attention)\n$\\rightarrow z^{(S)} \\in \\mathbb{R}^{96}$")

# Converging arrows
draw_arrow(ax, 0.275, 0.38, 0.42, 0.29)
draw_arrow(ax, 0.725, 0.38, 0.58, 0.29)

# --- 5. FEATURE FUSION ---
ax.text(0.5, 0.31, "5. FEATURE FUSION", ha='center', va='center', fontsize=11, fontweight='bold')
draw_box(ax, 0.3, 0.24, 0.4, 0.05, "Concatenation $\\rightarrow$ 2-Layer MLP")

draw_arrow(ax, 0.5, 0.24, 0.5, 0.18)

# --- 6. PREDICTION & VALIDATION ---
ax.text(0.5, 0.19, "6. PREDICTION & VALIDATION", ha='center', va='center', fontsize=11, fontweight='bold')
draw_box(ax, 0.15, 0.07, 0.7, 0.10, "Sigmoid $\\rightarrow$ Regional Infiltration Probability $\\hat{y}_i \\in [0, 1]$\n$\\downarrow$\nOptimization: BCE Loss w/ Logits + AdamW")

# Output Saving
out_dir = "e:/Zombozomes - research/paper"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "Full_Methodology_Pipeline.png")
pdf_path = os.path.join(out_dir, "Full_Methodology_Pipeline.pdf")
plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0.02)
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
print("Saved methodology diagram to", png_path)
