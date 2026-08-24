import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

batch = np.array([512, 1024, 2048, 4096, 8192])

# 数据
ours_10   = np.array([120129,135754,154710,166867,184229])
cabana_10 = np.array([58174,81997,105260,128088,158394])
faiss_10  = np.array([94215,82412,83828,78414,76445])

ours_32   = np.array([35619,44433,58537,70116,79919])
cabana_32 = np.array([22055,34798,49555,62748,77909])
faiss_32  = np.array([5785,5956,7592,7919,8082])

ours_all   = np.concatenate((ours_10, ours_32))
cabana_all = np.concatenate((cabana_10, cabana_32))
faiss_all  = np.concatenate((faiss_10, faiss_32))

speedup_vs_faiss  = ours_all / faiss_all
speedup_vs_cabana = ours_all / cabana_all

max_vs_faiss = np.max(speedup_vs_faiss)
avg_vs_faiss = np.mean(speedup_vs_faiss)

max_vs_cabana = np.max(speedup_vs_cabana)
avg_vs_cabana = np.mean(speedup_vs_cabana)

print("===== Overall Speedup (combined) =====")
print(f"Ours vs Faiss        : max = {max_vs_faiss:.3f}x, avg = {avg_vs_faiss:.3f}x")
print(f"Ours vs Faiss+CABANA: max = {max_vs_cabana:.3f}x, avg = {avg_vs_cabana:.3f}x")

x = np.arange(len(batch))
width = 0.25

# 🎨 颜色和方法绑定
color_faiss  = '#4C72B0'
color_cabana = '#55A868'
color_ours   = '#C44E52'

fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))

font1 = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 14.5,
}

font = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 14,
}

# ===== 左图 =====
ax = axs[0]

l1 = ax.bar(x - width, faiss_10,  width, color=color_faiss,  edgecolor='black')
l2 = ax.bar(x,         cabana_10, width, color=color_cabana, edgecolor='black')
l3 = ax.bar(x + width, ours_10,   width, color=color_ours,   edgecolor='black')

ax.set_title('Recall@10=0.90', font1)
ax.set_xlabel('Batch size', font1)
ax.set_ylabel('QPS', font1)
ax.set_xlim(-0.5, 4.5)
ax.set_yticks((0, 50000, 100000, 150000, 200000))
ax.set_ylim(0, 210000)
ax.set_yticklabels(("0", "50K", "100K", "150K", "200K"), fontsize=13.5)
ax.set_xticks(x)
ax.set_xticklabels(("512", "1024", "2048", "4096", "8192"), fontsize=13.5)
ax.grid(False)

ax.tick_params(axis='x', which='both', length=0)

# ===== 右图 =====
ax = axs[1]

ax.bar(x - width, faiss_32,  width, color=color_faiss,  edgecolor='black')
ax.bar(x,         cabana_32, width, color=color_cabana, edgecolor='black')
ax.bar(x + width, ours_32,   width, color=color_ours,   edgecolor='black')

ax.set_title('Recall@32=0.95', font1)
ax.set_xlabel('Batch size', font1)
ax.set_xlim(-0.5, 4.5)
ax.set_yticks((0, 20000, 40000, 60000, 80000))
ax.set_ylim(0, 81000)
ax.set_yticklabels(("0", "20K", "40K", "60K", "80K"), fontsize=13.5)
ax.set_xticks(x)
ax.set_xticklabels(("512", "1024", "2048", "4096", "8192"), fontsize=13.5)
ax.grid(False)

fig.legend(
    [l1, l2, l3],
    ['Faiss', 'Faiss+CABANA', 'Faiss+TILEFLOW'],
    ncol=3,
    loc="upper right",
    borderaxespad=0.0,
    bbox_to_anchor=(0.59, 1.08),
    columnspacing=0.5,
    frameon=False,
    prop=font,
    handlelength=0.8,
    handletextpad=0.1
)

ax.tick_params(axis='x', which='both', length=0)
plt.tight_layout()
plt.savefig('anns.pdf', bbox_inches='tight')
plt.show()