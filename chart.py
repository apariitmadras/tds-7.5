"""
chart.py — Professional Seaborn lineplot (512x512 PNG)

What this script does
- Generates realistic synthetic monthly revenue with clear seasonality for 3 product lines
- Plots a professional Seaborn lineplot with proper styling and labels
- Saves exactly 512x512 PNG at ./chart.png (verifies & fixes size if needed)

How to run
    pip install seaborn matplotlib pandas numpy
    python chart.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

# -------------------------
# 1) Generate synthetic data (seasonal revenue patterns)
# -------------------------
rng = np.random.default_rng(42)

months = pd.date_range("2023-01-01", periods=24, freq="MS")  # 2 years monthly
products = ["Engine Parts", "Body Components", "Electronics"]

rows = []
for product in products:
    base = {"Engine Parts": 420, "Body Components": 360, "Electronics": 300}[product]  # $K
    trend = {"Engine Parts": 3.0, "Body Components": 2.0, "Electronics": 4.0}[product]  # up per month
    # Seasonality: peak in Nov-Dec (index 10-11), softer in May-Jun
    seasonal = np.array([
        0.95, 0.98, 1.00, 0.98, 0.94, 0.93, 0.95, 0.99, 1.03, 1.06, 1.12, 1.15
    ])
    for i, m in enumerate(months):
        s = seasonal[m.month - 1]
        noise = rng.normal(0, 18)  # business noise in $K
        revenue_k = base * s + trend * i + noise
        rows.append({"month": m, "product": product, "revenue_k": max(revenue_k, 0)})

df = pd.DataFrame(rows)

# -------------------------
# 2) Seaborn styling (best practices)
# -------------------------
sns.set_style("whitegrid")
sns.set_context("talk")  # presentation-ready text sizes

# -------------------------
# 3) Create the lineplot
# -------------------------
plt.figure(figsize=(8, 8))  # 8 in * 64 dpi = 512 px
ax = sns.lineplot(
    data=df,
    x="month",
    y="revenue_k",
    hue="product",
    style="product",
    markers=True,
    dashes=False,
    linewidth=2.5,
    palette="tab10"
)

# Labels & Title
ax.set_title("Monthly Revenue by Product Line (Seasonality)", pad=12)
ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($K)")

# Format y-axis as $K
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"${v:,.0f}K"))

# Make month ticks readable
ax.set_xticks(df["month"].unique()[::2])  # every 2nd month
ax.tick_params(axis="x", rotation=45)

ax.legend(title="Product Line", frameon=True)
plt.tight_layout()

# -------------------------
# 4) Save exactly 512x512 PNG
# -------------------------
# Guideline: use bbox_inches='tight' with this dpi to target 512x512
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
plt.close()

# If bbox_inches changes the pixel size slightly, correct to 512x512.
with Image.open("chart.png") as im:
    if im.size != (512, 512):
        # Center on a 512x512 white canvas to preserve aspect without distortion
        canvas = Image.new("RGB", (512, 512), "white")
        im_thumb = im.copy()
        im_thumb.thumbnail((512, 512), Image.LANCZOS)
        x = (512 - im_thumb.width) // 2
        y = (512 - im_thumb.height) // 2
        canvas.paste(im_thumb, (x, y))
        canvas.save("chart.png")
