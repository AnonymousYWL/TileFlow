import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean


def read_data(file):
    data = []
    with open(file, 'r') as f:
        lines = f.read().split("\n")

    for item in lines:
        arr = item.split(" ")
        if len(arr) != 0 and arr[0] != '':
            mat = np.array([
                float(arr[-1]),
                float(arr[-2]),
                float(arr[-3]),
                float(arr[-4]),
                float(arr[-5])
            ])
            data.append(np.mean(mat))

    return data


# ============================================
# 6430
# ============================================

y1_6430 = read_data('6430/onednn_NN_INT8_6430.txt')
y2_6430 = read_data('6430/INT8_base_6430.txt')
y4_6430 = read_data('6430/GEMMs_Tileflow_INT32_6430.txt')


# ============================================
# 6448H
# ============================================

y1_6448H = read_data('6448H/onednn_NN_INT8_6448H.txt')
y2_6448H = read_data('6448H/INT8_base_6448H.txt')
y4_6448H = read_data('6448H/GEMMs_Tileflow_INT32_6448H.txt')


# ============================================
# Overall geometric mean speedup
# 6430 + 6448H together
# ============================================

speedup_vs_onednn = np.concatenate([
    np.array(y4_6430) / np.array(y1_6430),
    np.array(y4_6448H) / np.array(y1_6448H)
])

speedup_vs_mkl = np.concatenate([
    np.array(y4_6430) / np.array(y2_6430),
    np.array(y4_6448H) / np.array(y2_6448H)
])


print("{:.2f}x".format(gmean(speedup_vs_onednn)))
print("{:.2f}x".format(gmean(speedup_vs_mkl)))


# ============================================
# Plot
# ============================================

fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.5))

x = [32 + 32*i for i in range(len(y1_6430))]


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


# ===== subplot 1 : 6430 =====

l1, = axs[0].plot(
    x, y1_6430,
    color=(76/255,114/255,176/255),
    marker="D",
    markersize=3.5
)

l2, = axs[0].plot(
    x, y2_6430,
    color=(85/255,168/255,104/255),
    marker="^",
    markersize=3.5
)

l4, = axs[0].plot(
    x, y4_6430,
    color=(214/255,39/255,40/255),
    marker="o",
    markersize=3.5
)


axs[0].set_xlabel(r"$M$ (On 6430)", font1)
axs[0].set_ylabel("TOPS (INT8)", font)
axs[0].set_xlim(0,1050)
axs[0].grid(True, c="lightgrey", linestyle='--')
axs[0].set_xticks(
    (200,400,600,800,1000),
    ('200','400','600','800','1000'),
    fontsize=14
)
axs[0].set_yticks(
    (0,20,40,60,80),
    ('0','20','40','60','80'),
    fontsize=14
)
axs[0].set_ylim(0,82)
axs[0].tick_params(direction='in')

plt.setp(axs[0].get_xticklabels(), fontsize=15)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=15)



# ===== subplot 2 : 6448H =====

l1, = axs[1].plot(
    x, y1_6448H,
    color=(76/255,114/255,176/255),
    marker="D",
    markersize=3.5
)

l2, = axs[1].plot(
    x, y2_6448H,
    color=(85/255,168/255,104/255),
    marker="^",
    markersize=3.5
)

l4, = axs[1].plot(
    x, y4_6448H,
    color=(214/255,39/255,40/255),
    marker="o",
    markersize=3.5
)


axs[1].set_xlabel(r"$M$ (On 6448H)", font1)
axs[1].set_xlim(0,1050)
axs[1].grid(True, c="lightgrey", linestyle='--')
axs[1].set_xticks(
    (200,400,600,800,1000),
    ('200','400','600','800','1000'),
    fontsize=14
)
axs[1].set_yticks(
    (0,20,40,60,80),
    ('0','20','40','60','80'),
    fontsize=14
)
axs[1].set_ylim(0,90)
axs[1].tick_params(direction='in')

plt.setp(axs[1].get_xticklabels(), fontsize=15)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=15)



fig.legend(
    [l1,l2,l4],
    ['oneDNN','MKL','TILEFLOW'],
    ncol=3,
    loc="upper right",
    borderaxespad=0.0,
    bbox_to_anchor=(0.53,1.036),
    columnspacing=0.5,
    frameon=False,
    prop=font1,
    handlelength=1.5,
    handletextpad=0.1
)


fig.tight_layout()
plt.subplots_adjust(wspace=0.16, hspace=0.34)

plt.show()

fig.savefig('INT8.pdf', bbox_inches='tight')