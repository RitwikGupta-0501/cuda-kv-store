import matplotlib.pyplot as plt
import numpy as np
import os

# Data from benchmark
load_factors = [10, 20, 30, 40, 45, 50, 60, 70, 80, 90]
insert_throughput = [70.52, 73.54, 75.11, 76.11, 24.95, 15.19, 7.51, 4.11, 2.86, 2.08]
lookup_throughput = [92.14, 93.18, 94.25, 96.32, 87.29, 18.62, 18.40, 18.40, 17.88, 15.58]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(load_factors, insert_throughput, marker='o', linewidth=2, color='#00ffcc', label='Insert Throughput')
ax.plot(load_factors, lookup_throughput, marker='s', linewidth=2, color='#ff00ff', label='Lookup Throughput')

# Add a vertical line at 50% to show where rehashing should trigger
ax.axvline(x=50, color='#ff3333', linestyle='--', linewidth=2, label='Automatic Rehash Trigger (50%)')

ax.set_title('WarpKV Throughput vs. Load Factor (2-Way Cuckoo Hash)', fontsize=16, pad=20, color='white')
ax.set_xlabel('Load Factor (%)', fontsize=12, color='lightgray')
ax.set_ylabel('Throughput (Million keys/sec)', fontsize=12, color='lightgray')
ax.grid(True, linestyle=':', alpha=0.3)
ax.legend(fontsize=12, facecolor='black', edgecolor='gray')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'benchmark_graph.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Graph saved to {output_path}")
