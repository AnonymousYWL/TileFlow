import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean


# ==========================================================
# Utility Functions
# ==========================================================


def half_speedup(megemm, baseline):
    ratios = [a / b for a, b in zip(megemm, baseline)]
    mid = len(ratios) // 2
    return gmean(ratios[:mid]), gmean(ratios[mid:])

def load_data(filename):
    """
    Read benchmark file.

    Return:
        means : average TFLOPS
        stds  : standard deviation
    """
    means = []
    stds = []

    with open(filename, "r") as f:
        for line in f:
            arr = line.split()

            if len(arr) == 0:
                continue

            vals = np.array(list(map(float, arr[-5:])))

            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1))

    return means, stds


def print_stats(name, x, means, stds):
    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)

    cvs = []

    for m, mean, std in zip(x, means, stds):
        cv = std / mean * 100.0
        cvs.append(cv)

        print(
            f"M={m:4d} : "
            f"{mean:8.4f} ± {std:.4f} "
            f"(CV={cv:.2f}%)"
        )

    avg_mean = np.mean(means)
    avg_std = np.mean(stds)

    # 每个点CV的平均
    avg_cv = np.mean(cvs)

    # 整体标准差占整体均值
    overall_cv = avg_std / avg_mean * 100.0

    print("-" * 80)
    print(f"Average Mean = {avg_mean:.4f}")
    print(f"Average Std  = {avg_std:.4f}")
    print(f"Average CV   = {avg_cv:.2f}%")
    print(f"Std/Mean     = {overall_cv:.2f}%")
    print("-" * 80)


# ==========================================================
# Figure Setup
# ==========================================================

fig, axs = plt.subplots(1, 4, figsize=(15.2, 2.5))

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

colors = {
    "oneDNN": (46/255,117/255,182/255),
    "MKL": (112/255,173/255,71/255),
    "OpenBLAS": (240/255,131/255,58/255),
    "MeGEMM": (214/255,39/255,40/255),
}

markers = {
    "oneDNN": "D",
    "MKL": "^",
    "OpenBLAS": "s",
    "MeGEMM": "o",
}


# ==========================================================
# Subplot 1
# NN BF16-BF16
# ==========================================================

y1, y1_std = load_data("onednn_NN_BF16_6430.txt")
y4, y4_std = load_data("MeGEMM_NN_BF16_6430.txt")

x = [32 + 32*i for i in range(len(y1))]

print_stats("oneDNN NN BF16-BF16", x, y1, y1_std)
print_stats("MeGEMM NN BF16-BF16", x, y4, y4_std)

l1, = axs[0].plot(
    x, y1,
    color=colors["oneDNN"],
    marker=markers["oneDNN"],
    markersize=3.5
)

l4, = axs[0].plot(
    x, y4,
    color=colors["MeGEMM"],
    marker=markers["MeGEMM"],
    markersize=3.5
)

axs[0].set_xlabel(r"$M\ $ (NN mode, BF16-BF16)", font1)
axs[0].set_ylabel("TFLOPS (BF16)", font)

speedup_1 = gmean([a/b for a, b in zip(y4, y1)])
print("\nSubplot1 MeGEMM vs oneDNN Speedup:", speedup_1)


# ==========================================================
# Subplot 2
# NN BF16-FP32
# ==========================================================

y1, y1_std = load_data("onednn_NN_FP32_6430.txt")
y2, y2_std = load_data("MKL_NN_FP32_6430.txt")
y3, y3_std = load_data("openblas_NN_FP32_6430.txt")
y4, y4_std = load_data("MeGEMM_NN_FP32_6430.txt")

print_stats("oneDNN NN BF16-FP32", x, y1, y1_std)
print_stats("MKL NN BF16-FP32", x, y2, y2_std)
print_stats("OpenBLAS NN BF16-FP32", x, y3, y3_std)
print_stats("MeGEMM NN BF16-FP32", x, y4, y4_std)

l1, = axs[1].plot(
    x, y1,
    color=colors["oneDNN"],
    marker=markers["oneDNN"],
    markersize=3.5
)

l2, = axs[1].plot(
    x, y2,
    color=colors["MKL"],
    marker=markers["MKL"],
    markersize=3.5
)

l3, = axs[1].plot(
    x, y3,
    color=colors["OpenBLAS"],
    marker=markers["OpenBLAS"],
    markersize=3.5
)

l4, = axs[1].plot(
    x, y4,
    color=colors["MeGEMM"],
    marker=markers["MeGEMM"],
    markersize=3.5
)

axs[1].set_xlabel(r"$M\ $ (NN mode, BF16-FP32)", font1)

speedup_21 = gmean([a/b for a, b in zip(y4, y1)])
speedup_22 = gmean([a/b for a, b in zip(y4, y2)])
speedup_23 = gmean([a/b for a, b in zip(y4, y3)])

print("\nSubplot2 MeGEMM vs oneDNN Speedup:", speedup_21)
print("Subplot2 MeGEMM vs MKL Speedup:", speedup_22)
print("Subplot2 MeGEMM vs OpenBLAS Speedup:", speedup_23)

first_half_22, second_half_22 = half_speedup(y4, y2)

print(
    f"Subplot2 MeGEMM vs MKL (First Half)  Speedup: "
    f"{first_half_22:.4f}x"
)

print(
    f"Subplot2 MeGEMM vs MKL (Second Half) Speedup: "
    f"{second_half_22:.4f}x"
)


# ==========================================================
# Subplot 3
# NT BF16-BF16
# ==========================================================

y1, y1_std = load_data("onednn_NT_BF16_6430.txt")
y4, y4_std = load_data("MeGEMM_NT_BF16_6430.txt")

print_stats("oneDNN NT BF16-BF16", x, y1, y1_std)
print_stats("MeGEMM NT BF16-BF16", x, y4, y4_std)

l1, = axs[2].plot(
    x, y1,
    color=colors["oneDNN"],
    marker=markers["oneDNN"],
    markersize=3.5
)

l4, = axs[2].plot(
    x, y4,
    color=colors["MeGEMM"],
    marker=markers["MeGEMM"],
    markersize=3.5
)

axs[2].set_xlabel(r"$M\ $ (NT mode, BF16-BF16)", font1)

speedup_3 = gmean([a/b for a, b in zip(y4, y1)])

print("\nSubplot3 MeGEMM vs oneDNN Speedup:", speedup_3)


# ==========================================================
# Subplot 4
# NT BF16-FP32
# ==========================================================

y1, y1_std = load_data("onednn_NT_FP32_6430.txt")
y2, y2_std = load_data("MKL_NT_FP32_6430.txt")
y3, y3_std = load_data("openblas_NT_FP32_6430.txt")
y4, y4_std = load_data("MeGEMM_NT_FP32_6430.txt")

print_stats("oneDNN NT BF16-FP32", x, y1, y1_std)
print_stats("MKL NT BF16-FP32", x, y2, y2_std)
print_stats("OpenBLAS NT BF16-FP32", x, y3, y3_std)
print_stats("MeGEMM NT BF16-FP32", x, y4, y4_std)

l1, = axs[3].plot(
    x, y1,
    color=colors["oneDNN"],
    marker=markers["oneDNN"],
    markersize=3.5
)

l2, = axs[3].plot(
    x, y2,
    color=colors["MKL"],
    marker=markers["MKL"],
    markersize=3.5
)

l3, = axs[3].plot(
    x, y3,
    color=colors["OpenBLAS"],
    marker=markers["OpenBLAS"],
    markersize=3.5
)

l4, = axs[3].plot(
    x, y4,
    color=colors["MeGEMM"],
    marker=markers["MeGEMM"],
    markersize=3.5
)

axs[3].set_xlabel(r"$M\ $ (NT mode, BF16-FP32)", font1)

speedup_41 = gmean([a/b for a, b in zip(y4, y1)])
speedup_42 = gmean([a/b for a, b in zip(y4, y2)])
speedup_43 = gmean([a/b for a, b in zip(y4, y3)])

print("\nSubplot4 MeGEMM vs oneDNN Speedup:", speedup_41)
print("Subplot4 MeGEMM vs MKL Speedup:", speedup_42)
print("Subplot4 MeGEMM vs OpenBLAS Speedup:", speedup_43)


# ==========================================================
# Common Axis Style
# ==========================================================

for i in range(4):
    axs[i].set_xlim(0, 1050)
    axs[i].grid(True, c="lightgrey", linestyle='--')
    axs[i].set_xticks(
        (200,400,600,800,1000),
        ('200','400','600','800','1000'),
        fontsize=14
    )
    axs[i].tick_params(direction='in')
    plt.setp(axs[i].get_xticklabels(), fontsize=15)

axs[0].set_ylim(0, 42.5)
axs[0].set_yticks((0,10,20,30,40))
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=15)

axs[1].set_ylim(0, 40.5)
axs[1].set_yticks((0,10,20,30,40))
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=15)

axs[2].set_ylim(0, 40.5)
axs[2].set_yticks((0,10,20,30,40))
axs[2].set_yticklabels(axs[2].get_yticks(), fontsize=15)

axs[3].set_ylim(0, 36.5)
axs[3].set_yticks((0,10,20,30,35))
axs[3].set_yticklabels(axs[3].get_yticks(), fontsize=15)


first_half_42, second_half_42 = half_speedup(y4, y2)

print(
    f"Subplot4 MeGEMM vs MKL (First Half)  Speedup: "
    f"{first_half_42:.4f}x"
)

print(
    f"Subplot4 MeGEMM vs MKL (Second Half) Speedup: "
    f"{second_half_42:.4f}x"
)


# ==========================================================
# Legend
# ==========================================================

fig.legend(
    [l1, l2, l3, l4],
    ['oneDNN', 'MKL', 'OpenBLAS', 'TILEFLOW'],
    ncol=4,
    loc="upper right",
    borderaxespad=0.0,
    bbox_to_anchor=(0.355,1.03),
    columnspacing=0.5,
    frameon=False,
    prop=font1,
    handlelength=1.5,
    handletextpad=0.1
)


# ==========================================================
# Overall Speedup
# ==========================================================

print("\n")
print("=" * 80)
print("Overall Geometric Mean Speedup")
print("=" * 80)

print("vs oneDNN :", gmean([speedup_1, speedup_21, speedup_3, speedup_41]))
print("vs MKL    :", gmean([speedup_22, speedup_42]))
print("vs OpenBLAS :", gmean([speedup_23, speedup_43]))



overall_first_half = gmean([
    first_half_22,
    first_half_42
])

overall_second_half = gmean([
    second_half_22,
    second_half_42
])

print()
print("=" * 80)
print("MeGEMM vs MKL Half-wise Speedup")
print("=" * 80)
print(f"First Half  : {overall_first_half:.4f}x")
print(f"Second Half : {overall_second_half:.4f}x")


# ==========================================================
# Save Figure
# ==========================================================

fig.tight_layout()
plt.subplots_adjust(wspace=0.16, hspace=0.34)

plt.show()

fig.savefig(
    "GEMM_6430.pdf",
    bbox_inches="tight"
)