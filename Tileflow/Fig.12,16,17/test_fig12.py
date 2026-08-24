import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean


def read_mean(filename):
    y = []
    with open(filename, 'r') as f:
        for line in f:
            arr = line.split()
            if arr:
                mat = np.array(list(map(float, arr[-5:])))
                y.append(np.mean(mat))
    return y


y1 = read_mean('../UMA/MKL_NT_FP32_6430_table_UMA.txt')
y2 = read_mean('../UMA/onednn_NT_FP32_6430_table_UMA.txt')
y3 = read_mean('../UMA/OpenBLAS_NT_FP32_6430_table_UMA.txt')
y4 = read_mean('../UMA/MeGEMM_NT_FP32_6430_UMA.txt')

y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)

# speedup
ratio2 = y2 / y1
ratio3 = y3 / y1
ratio4 = y4 / y1

geo_mean2 = gmean(ratio2)
geo_mean3 = gmean(ratio3)
geo_mean4 = gmean(ratio4)

print("oneDNN 几何平均:", geo_mean2)
print("OpenBLAS 几何平均:", geo_mean3)
print("MeGEMM 几何平均:", geo_mean4)

max_val_uma = np.max(ratio4)
max_idx_uma = np.argmax(ratio4)
print(f"MeGEMM ratio4 最大值 (UMA): {max_val_uma:.4f}, 出现在索引: {max_idx_uma}")

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 15}

data = [ratio2,ratio3,ratio4]

vp = axs[0].violinplot(
    data,
    showmeans=True,
    showmedians=True,
    showextrema=True
)

# 论文常用颜色
colors = ["#2E75B6", "#70AD47", "#C44E52"]

for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

# x轴
axs[0].set_xticks([1,2,3])
axs[0].set_xticklabels(['oneDNN', 'OpenBLAS', 'TILEFLOW'], fontdict=font)

# y轴
axs[0].set_ylabel('Normalized Speedup', fontdict=font, labelpad=15)

# baseline
axs[0].set_ylabel('Normalized Speedup', fontdict=font)
axs[0].yaxis.set_label_coords(-0.13, 0.46)
axs[0].set_ylim(0, 2.05)
axs[0].set_yticks([0, 0.5, 1.0, 1.5, 2.0])

# 几何平均点
axs[0].scatter([1,2,3],
               [geo_mean2, geo_mean3, geo_mean4],
               color='black',
               s=25,
               zorder=3)

axs[0].grid(axis='y', linestyle='--', alpha=0.5)

plt.setp(axs[0].get_xticklabels(),fontsize=13)
axs[0].tick_params(axis='y', labelsize=13)
axs[0].tick_params(direction='in')

axs[0].axhline(
    y=1,
    color='black',
    linestyle='--',
    linewidth=1.1,
    alpha=0.5
)

axs[0].set_title("Single Socket", fontsize=15)
axs[1].set_title("Dual Socket", fontsize=15)


y1 = read_mean('MKL_NT_FP32_6430_table.txt')
y2 = read_mean('onednn_NT_FP32_6430_table.txt')
y3 = read_mean('OpenBLAS_NT_FP32_6430_table.txt')
y4 = read_mean('MeGEMM_NT_FP32_6430_table.txt')

y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)

# speedup
ratio2 = y2 / y1
ratio3 = y3 / y1
ratio4 = y4 / y1

geo_mean2 = gmean(ratio2)
geo_mean3 = gmean(ratio3)
geo_mean4 = gmean(ratio4)

print("oneDNN 几何平均:", geo_mean2)
print("OpenBLAS 几何平均:", geo_mean3)
print("MeGEMM 几何平均:", geo_mean4)

max_val_dual = np.max(ratio4)
max_idx_dual = np.argmax(ratio4)
print(f"MeGEMM ratio4 最大值 (Dual): {max_val_dual:.4f}, 出现在索引: {max_idx_dual}")

data = [ratio2, ratio3, ratio4]

vp = axs[1].violinplot(
    data,
    showmeans=True,
    showmedians=True,
    showextrema=True
)

for i, pc in enumerate(vp['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

# x轴
axs[1].set_xticks([1,2,3])
axs[1].set_xticklabels(['oneDNN', 'OpenBLAS', 'TILEFLOW'], fontdict=font)

# 几何平均点
axs[1].scatter([1,2,3],
               [geo_mean2, geo_mean3, geo_mean4],
               color='black',
               s=25,
               zorder=3)

axs[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.setp(axs[1].get_xticklabels(),fontsize=13)
axs[1].tick_params(axis='y', labelsize=13)
axs[1].tick_params(direction='in')

axs[1].set_ylim(0, 2.7)
axs[1].set_yticks([0, 0.5, 1.0, 1.5, 2.0,2.5])

axs[1].axhline(
    y=1,
    color='black',
    linestyle='--',
    linewidth=1.1,
    alpha=0.5
)


# =====================================================
# TILEFLOW 相对 MKL / oneDNN / OpenBLAS 的加速比分布统计
# =====================================================

def calc_percentage(ratio):
    total = len(ratio)

    less_equal_1 = np.sum(ratio <= 1.0) / total * 100
    p1 = np.sum((ratio > 1.0) & (ratio <= 1.5)) / total * 100
    p2 = np.sum((ratio > 1.5) & (ratio <= 2.0)) / total * 100
    p3 = np.sum((ratio > 2.0) & (ratio <= 2.5)) / total * 100
    p4 = np.sum(ratio > 2.5) / total * 100

    return less_equal_1, p1, p2, p3, p4


# TILEFLOW 相对各库
ratio_tf_mkl = y4 / y1
ratio_tf_onednn = y4 / y2
ratio_tf_openblas = y4 / y3

print("\n================ Dual Socket Statistics ================")

for name, ratio in [
    ("MKL", ratio_tf_mkl),
    ("oneDNN", ratio_tf_onednn),
    ("OpenBLAS", ratio_tf_openblas)
]:
    less1, p1, p2, p3, p4 = calc_percentage(ratio)

    print(f"\nTILEFLOW vs {name}")
    print(f"<=1        : {less1:.2f}%")
    print(f"(1,1.5]    : {p1:.2f}%")
    print(f"(1.5,2]    : {p2:.2f}%")
    print(f"(2,2.5]    : {p3:.2f}%")
    print(f">2.5       : {p4:.2f}%")
    print(f"Check Sum  : {less1+p1+p2+p3+p4:.2f}%")


target = 2.64

idx = np.argmin(np.abs(ratio4 - target))

print("\n最接近 2.64 的 case:")
print(f"Index = {idx}")
print(f"Speedup = {ratio4[idx]:.6f}")
print(f"MKL      = {y1[idx]:.6f}")
print(f"TILEFLOW = {y4[idx]:.6f}")

fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace =0.16, hspace = 0.34)
plt.show()
fig.savefig('Violin.pdf', bbox_inches='tight')