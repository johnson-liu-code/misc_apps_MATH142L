import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.axis('off')

col_labels = ['H₀ True', 'H₀ False']
row_labels = ['Reject H₀', 'Fail to\nReject H₀']
cell_text = [['False\nPositive', 'True\nPositive'],
             ['True\nNegative', 'False\nNegative']]
cell_colors = [[(1,0,0,0.4), (0,1,0,0.3)],
               [(0,0.784,1,0.3), (0.502,0,1,0.3)]]

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fill=True, edgecolor='black', facecolor=cell_colors[i][j]))
        ax.text(j + 0.5, 1.5 - i, cell_text[i][j], va='center', ha='center', fontsize=12, weight='bold')

for j, label in enumerate(col_labels):
    ax.text(j + 0.5, 2.05, label, ha='center', va='center', fontsize=14, weight='bold')

for i, label in enumerate(row_labels):
    ax.text(-0.1, 1.5 - i, label, ha='right', va='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches='tight')
plt.show()
