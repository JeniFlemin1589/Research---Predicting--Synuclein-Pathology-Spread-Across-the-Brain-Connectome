import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(6, 8.5))
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1.05)
ax.axis('off')

# Use bbox to naturally overwrite any potential tiny line overlaps, adding to clarity
def add_text(x, y, text, size=11, weight='normal', style='normal', color='black'):
    ax.text(x, y, text, ha='center', va='center', fontsize=size, fontweight=weight, fontstyle=style, color=color,
            bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none'))

def add_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color='black'))

# --- Coordinates ---
# 1. INPUT
add_text(0.5, 1.00, "1. INPUT MODULE", size=11, weight='bold')
add_text(0.5, 0.95, r"$\alpha$-Synuclein Pathology Data $\quad+\quad$ Brain Connectome $G$", size=11)

add_arrow(0.5, 0.92, 0.5, 0.85)

# 2. PREP
add_text(0.5, 0.82, "2. DATA & GRAPH PREPROCESSING", size=11, weight='bold')
add_text(0.5, 0.77, r"Kinetic Traces $(T=\{1,3,6\}) \longrightarrow$ Temporal Aggregation $(z, \Delta)$", size=11)
add_text(0.5, 0.72, r"Connectome $G \longrightarrow$ Structural Edges & Vulnerability Features", size=11)

add_arrow(0.5, 0.69, 0.5, 0.63)

# 3. ARCHITECTURE
add_text(0.5, 0.60, "3. DUAL-BRANCH ARCHITECTURE", size=11, weight='bold')

# Split arrows
add_arrow(0.5, 0.57, 0.25, 0.51)
add_arrow(0.5, 0.57, 0.75, 0.51)

# Branches
add_text(0.25, 0.47, "Temporal Branch", size=11, style='italic')
add_text(0.25, 0.42, "GRU Sequence Modeling", size=11)
add_text(0.25, 0.36, r"$z^{(T)} \in \mathbb{R}^{96}$", size=12)

add_text(0.75, 0.47, "Spatial Branch", size=11, style='italic')
add_text(0.75, 0.42, "GATv2 Contextual Attention", size=11)
add_text(0.75, 0.36, r"$z^{(S)} \in \mathbb{R}^{96}$", size=12)

# Merge arrows
add_arrow(0.25, 0.32, 0.47, 0.24)
add_arrow(0.75, 0.32, 0.53, 0.24)

# 4. FUSION
add_text(0.5, 0.20, "4. FEATURE FUSION", size=11, weight='bold')
add_text(0.5, 0.15, r"Concatenate $z^{(T)} \parallel z^{(S)} \longrightarrow$ 2-Layer MLP", size=11)

add_arrow(0.5, 0.12, 0.5, 0.07)

# 5. PREDICTION
add_text(0.5, 0.04, "5. PREDICTION & VALIDATION", size=11, weight='bold')
add_text(0.5, -0.01, r"Sigmoid $\longrightarrow$ Regional Infiltration Probability $\hat{y}_i \in [0, 1]$", size=11)
add_text(0.5, -0.06, r"Optimization $\longrightarrow$ BCE Loss + AdamW Updates", size=11)

out_dir = "e:/Zombozomes - research/paper"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "Full_Methodology_Pipeline_Clean.png")
pdf_path = os.path.join(out_dir, "Full_Methodology_Pipeline_Clean.pdf")
plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0.04)
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
print("Saved methodology diagram to", png_path)
