import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Set serif font
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(6, 7))
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1.05)
ax.axis('off')

def add_text(x, y, text, size=11, weight='normal', style='normal', color='black'):
    ax.text(x, y, text, ha='center', va='center', fontsize=size, fontweight=weight, fontstyle=style, color=color)

def add_arrow(x1, y1, x2, y2, rad="0.0"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color='black', connectionstyle=f"arc3,rad={rad}"))

# 1. Input
add_text(0.5, 1.0, "1. INPUT MODULE", size=10, weight='bold')
add_text(0.5, 0.95, r"$\alpha$-Synuclein Pathology Data $\quad+\quad$ Brain Connectome $G$", size=11)
add_arrow(0.5, 0.90, 0.5, 0.83)

# 2. Prep
add_text(0.5, 0.79, "2. DATA & GRAPH PREPROCESSING", size=10, weight='bold')
add_text(0.5, 0.74, r"Kinetic Traces $(T=\{1,3,6\}) \longrightarrow$ Temporal Aggregation $(z, \Delta)$", size=11)
add_text(0.5, 0.69, r"Connectome $G \longrightarrow$ Structural Edges & Vulnerability Features", size=11)

add_arrow(0.5, 0.64, 0.25, 0.56, rad="0.1")
add_arrow(0.5, 0.64, 0.75, 0.56, rad="-0.1")

# 3. Arch
add_text(0.5, 0.58, "3. DUAL-BRANCH ARCHITECTURE", size=10, weight='bold')

add_text(0.25, 0.52, "Temporal Branch", size=11, style='italic')
add_text(0.25, 0.47, "GRU Sequence Modeling", size=11)
add_text(0.25, 0.42, r"$z^{(T)} \in \mathbb{R}^{96}$", size=12)

add_text(0.75, 0.52, "Spatial Branch", size=11, style='italic')
add_text(0.75, 0.47, "GATv2 Contextual Attention", size=11)
add_text(0.75, 0.42, r"$z^{(S)} \in \mathbb{R}^{96}$", size=12)

add_arrow(0.25, 0.36, 0.42, 0.28, rad="-0.1")
add_arrow(0.75, 0.36, 0.58, 0.28, rad="0.1")

# 4. Fusion
add_text(0.5, 0.24, "4. FEATURE FUSION", size=10, weight='bold')
add_text(0.5, 0.19, r"Concatenate $z^{(T)} \parallel z^{(S)} \longrightarrow$ 2-Layer MLP", size=11)
add_arrow(0.5, 0.14, 0.5, 0.07)

# 5. Validation
add_text(0.5, 0.03, "5. PREDICTION & VALIDATION", size=10, weight='bold')
add_text(0.5, -0.02, r"Sigmoid $\longrightarrow$ Regional Infiltration Probability $\hat{y}_i \in [0, 1]$", size=11)
add_text(0.5, -0.07, r"Optimization $\longrightarrow$ BCE Loss + AdamW Updates", size=11)

out_dir = "e:/Zombozomes - research/paper"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "Full_Methodology_Pipeline_NoBoxes.png")
pdf_path = os.path.join(out_dir, "Full_Methodology_Pipeline_NoBoxes.pdf")
plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0.04)
plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
print("Saved methodology diagram to", png_path)
