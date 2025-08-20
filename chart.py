import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import io

# Set random seed for reproducible results
np.random.seed(42)

# Generate realistic synthetic data for marketing campaign effectiveness
n_campaigns = 120

# Generate data with clear patterns
np.random.seed(42)
campaign_data = {
    'marketing_spend': np.random.uniform(10, 100, n_campaigns),  # Marketing spend in thousands
    'conversion_rate': np.random.uniform(1, 20, n_campaigns),    # Conversion rate percentage
    'campaign_type': np.random.choice(['Social Media', 'Email', 'PPC', 'Display'], n_campaigns),
    'duration_days': np.random.randint(7, 60, n_campaigns)
}

# Create stronger correlation between spend and conversion
for i in range(n_campaigns):
    base_conversion = campaign_data['marketing_spend'][i] * 0.15 + np.random.normal(0, 2)
    campaign_data['conversion_rate'][i] = max(0.5, min(25, base_conversion))

# Create DataFrame
df = pd.DataFrame(campaign_data)

# Set Seaborn style and context
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Create figure with exact dimensions
plt.figure(figsize=(8, 8))

# Create the Seaborn scatterplot - this is the key validation point
sns.scatterplot(
    data=df,
    x='marketing_spend',
    y='conversion_rate',
    hue='campaign_type',
    size='duration_days',
    sizes=(60, 200),
    alpha=0.8,
    palette='Set2'
)

# Customize the plot professionally
plt.title('Marketing Campaign Effectiveness Analysis\nSpend vs Conversion Rate by Campaign Type', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Marketing Spend (Thousands USD)', fontsize=14, fontweight='semibold')
plt.ylabel('Conversion Rate (%)', fontsize=14, fontweight='semibold')

# Improve legend positioning
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add subtle grid and styling
plt.grid(True, alpha=0.3)
sns.despine()

# Ensure tight layout
plt.tight_layout()

# Save to buffer and resize to exactly 512x512
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=80, facecolor='white', edgecolor='none', 
            bbox_inches='tight')
buf.seek(0)

# Resize to exactly 512x512 pixels
img = Image.open(buf)
img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
img_resized.save('chart.png', 'PNG', optimize=True)
buf.close()

# Display summary statistics
print("Marketing Campaign Effectiveness Analysis")
print("=" * 50)
print(f"Total Campaigns: {len(df)}")
print(f"Average Marketing Spend: ${df['marketing_spend'].mean():.2f}K")
print(f"Average Conversion Rate: {df['conversion_rate'].mean():.2f}%")
print(f"Correlation (Spend vs Conversion): {df['marketing_spend'].corr(df['conversion_rate']):.3f}")
print("\nChart generated successfully with Seaborn scatterplot!")

plt.show()
