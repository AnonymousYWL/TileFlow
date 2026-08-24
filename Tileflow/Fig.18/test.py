import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean

# ============================================

f1 = open('openblas_SME.txt', 'r')
xy1 = f1.read().split("\n")

f2 = open('libxsmm_SME.txt', 'r')
xy2 = f2.read().split("\n")

f3 = open('accelerate_SME.txt', 'r')
xy3 = f3.read().split("\n")

f4 = open('megemm_SME.txt', 'r')
xy4 = f4.read().split("\n")

y1 = []
y2 = []
y3 = []
y4 = []
x = []

for item in xy1:
    arr = item.split(" ")
    if (len(arr) != 0 and arr[0] != ''):
        mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3])])
        y1.append(np.mean(mat) / 1000)

for item in xy2:
    arr = item.split(" ")
    if (len(arr) != 0 and arr[0] != ''):
        mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3])])
        y2.append(np.mean(mat) / 1000)

for item in xy3:
    arr = item.split(" ")
    if (len(arr) != 0 and arr[0] != ''):
        mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3])])
        y3.append(np.mean(mat) / 1000)

for item in xy4:
    arr = item.split(" ")
    if (len(arr) != 0 and arr[0] != ''):
        mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3])])
        y4.append(np.mean(mat) / 1000)

fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6))

x = []
for i in range(len(y1)):
    x.append(32 + 32 * i)

font1 = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 13.5,
}

font = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 15,
}

# ===== subplot 1 =====
l1, = axs[0].plot(x, y1, color=(76/255,114/255,176/255), marker="D", markersize=3.5)
l2, = axs[0].plot(x, y2, color=(85/255,168/255,104/255), marker="^", markersize=3.5)
l3, = axs[0].plot(x, y3, color=(240/255,131/255,58/255), marker="s", markersize=3.5)
l4, = axs[0].plot(x, y4, color=(214/255,39/255,40/255), marker="o", markersize=3.5)

# ===== Speedup (Single SME) =====
y1_arr = np.array(y1)
y2_arr = np.array(y2)
y3_arr = np.array(y3)
y4_arr = np.array(y4)

s41 = y4_arr / y1_arr
s42 = y4_arr / y2_arr
s43 = y4_arr / y3_arr

print("=== Single SME (Geometric Mean Speedup) ===")
print(f"TILEFLOW vs openBLAS: {gmean(s41):.2f}×")
print(f"TILEFLOW vs LIBXSMM: {gmean(s42):.2f}×")
print(f"TILEFLOW vs Accelerate: {gmean(s43):.2f}×")

axs[0].set_xlabel(r"$M$ (single SME unit)", font1)
axs[0].set_ylabel("TFLOPS (BF16)", font)
axs[0].set_xlim(0, 1050)
axs[0].grid(True, c="lightgrey", linestyle='--')
axs[0].set_xticks((200,400,600,800,1000), ('200','400','600','800','1000'), fontsize=14)
axs[0].set_yticks((0, 0.5,1.0,1.5), ('0','0.5','1.0','1.5'), fontsize=14)
axs[0].set_ylim(0, 1.6)
axs[0].tick_params(direction='in')

plt.setp(axs[0].get_xticklabels(), fontsize=15)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=15)

# ===== subplot 2 =====
y1 = []
y2 = []

f1 = open('accelerate_SMEs.txt', 'r')
xy1 = f1.read().split("\n")

f2 = open('megemm_SMEs.txt', 'r')
xy2 = f2.read().split("\n")

for item in xy1:
    arr = item.split(" ")
    if (len(arr) != 0 and arr[0] != ''):
        mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3])])
        y1.append(np.mean(mat) / 1000)

for item in xy2:
    arr = item.split(" ")
    if (len(arr) != 0 and arr[0] != ''):
        mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3])])
        y2.append(np.mean(mat) / 1000)

axs[1].plot(x, y1, color=(240/255,131/255,58/255), marker="D", markersize=3.5)
axs[1].plot(x, y2, color=(214/255,39/255,40/255), marker="o", markersize=3.5)

# ===== Speedup (Two SMEs) =====
y1_arr = np.array(y1)   # accelerate
y2_arr = np.array(y2)   # TILEFLOW

s21 = y2_arr / y1_arr

print("=== Two SMEs (Geometric Mean Speedup) ===")
print(f"TILEFLOW vs Accelerate: {gmean(s21):.2f}×")

axs[1].set_xlabel(r"$M$ (two SME units)", font1)
axs[1].set_xlim(0, 1050)
axs[1].grid(True, c="lightgrey", linestyle='--')
axs[1].set_xticks((200,400,600,800,1000), ('200','400','600','800','1000'), fontsize=14)
axs[1].set_yticks((0,1.0,2.0,3), ('0','1.0','2.0','3.0'), fontsize=14)
axs[1].set_ylim(0, 3.2)

plt.setp(axs[1].get_xticklabels(), fontsize=15)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=15)
axs[1].tick_params(direction='in')

fig.legend(
    [l1,l2,l3,l4],
    ['OpenBLAS','LIBXSMM','Accelerate','TILEFLOW'],
    ncol=4,
    loc="upper right",
    borderaxespad=0.0,
    bbox_to_anchor=(0.81,1.05),
    columnspacing=0.5,
    frameon=False,
    prop=font1,
    handlelength=1.5,
    handletextpad=0.1
)

fig.tight_layout()
plt.subplots_adjust(wspace=0.22, hspace=0.34)

plt.show()
fig.savefig('SME.pdf', bbox_inches='tight')