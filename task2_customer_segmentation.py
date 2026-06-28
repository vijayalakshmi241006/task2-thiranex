"""
Task 2 - Customer Segmentation using K-Means Clustering
Thiranex Data Analytics Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

# ── 1. Generate synthetic customer dataset ──────────────────────────────────
n = 300
data = {
    'CustomerID': range(1001, 1001 + n),
    'Age': np.random.randint(18, 70, n),
    'Annual_Income_K': np.random.randint(20, 150, n),
    'Spending_Score': np.random.randint(1, 100, n),
    'Purchase_Frequency': np.random.randint(1, 52, n),
    'Avg_Order_Value': np.round(np.random.uniform(200, 5000, n), 2),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
}
df = pd.DataFrame(data)

print("=" * 55)
print("  CUSTOMER SEGMENTATION - THIRANEX INTERNSHIP (TASK 2)")
print("=" * 55)
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))
print("\nBasic statistics:")
print(df.describe().round(2).to_string())

# ── 2. Feature Engineering & Scaling ───────────────────────────────────────
features = ['Age', 'Annual_Income_K', 'Spending_Score',
            'Purchase_Frequency', 'Avg_Order_Value']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Elbow Method + Silhouette Score ─────────────────────────────────────
inertias, silhouettes = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

best_k = k_range[np.argmax(silhouettes)]
print(f"\nOptimal number of clusters (max silhouette): {best_k}")

# ── 4. Final K-Means Model ─────────────────────────────────────────────────
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ── 5. Cluster Profiling ───────────────────────────────────────────────────
cluster_profile = df.groupby('Cluster')[features].mean().round(2)
cluster_profile['Size'] = df['Cluster'].value_counts().sort_index()

labels_map = {
    0: 'Budget Shoppers',
    1: 'High-Value Loyals',
    2: 'Young Explorers',
    3: 'Occasional Buyers',
    4: 'Premium Segment',
}
cluster_profile['Segment_Name'] = [labels_map.get(i, f'Segment {i}') for i in cluster_profile.index]

print("\nCluster Profiles:")
print(cluster_profile.to_string())

# ── 6. Visualisation ───────────────────────────────────────────────────────
colors = ['#378ADD', '#1D9E75', '#D85A30', '#BA7517', '#7F77DD']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold', y=1.01)

# 6a. Elbow + Silhouette
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(list(k_range), inertias, 'o-', color='#378ADD', label='Inertia')
ax2.plot(list(k_range), silhouettes, 's--', color='#1D9E75', label='Silhouette')
ax.axvline(best_k, color='#D85A30', linestyle=':', alpha=0.7, label=f'Best k={best_k}')
ax.set_xlabel('Number of clusters'); ax.set_ylabel('Inertia', color='#378ADD')
ax2.set_ylabel('Silhouette score', color='#1D9E75')
ax.set_title('Elbow Method & Silhouette Score')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
ax.grid(alpha=0.3)

# 6b. Scatter: Income vs Spending
ax = axes[0, 1]
for i in range(best_k):
    mask = df['Cluster'] == i
    ax.scatter(df[mask]['Annual_Income_K'], df[mask]['Spending_Score'],
               c=colors[i % len(colors)], label=cluster_profile.loc[i, 'Segment_Name'],
               alpha=0.65, s=40)
ax.set_xlabel('Annual Income (₹000)'); ax.set_ylabel('Spending Score')
ax.set_title('Income vs Spending Score by Segment')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 6c. Segment size bar
ax = axes[1, 0]
seg_sizes = cluster_profile['Size']
bars = ax.bar(cluster_profile['Segment_Name'], seg_sizes,
              color=colors[:best_k], edgecolor='white', linewidth=0.5)
ax.set_title('Segment Size Distribution')
ax.set_ylabel('Number of customers')
ax.set_xticks(range(best_k))
ax.set_xticklabels(cluster_profile['Segment_Name'], rotation=20, ha='right', fontsize=8)
for bar, val in zip(bars, seg_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 6d. Avg Order Value by segment
ax = axes[1, 1]
ax.barh(cluster_profile['Segment_Name'], cluster_profile['Avg_Order_Value'],
        color=colors[:best_k], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Avg Order Value (₹)'); ax.set_title('Average Order Value per Segment')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(cluster_profile['Avg_Order_Value']):
    ax.text(v + 20, i, f'₹{v:,.0f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task2_segmentation.png', dpi=150, bbox_inches='tight')
print("\nChart saved → task2_segmentation.png")

# ── 7. Save result CSV ────────────────────────────────────────────────────
df.to_csv('/mnt/user-data/outputs/task2_segmented_customers.csv', index=False)
print("Segmented data saved → task2_segmented_customers.csv")
print("\n✓ Task 2 complete!")
