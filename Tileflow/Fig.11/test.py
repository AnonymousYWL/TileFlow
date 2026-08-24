import numpy as np
import matplotlib.pyplot as plt

# ============================================

# 读取数据
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.read().split("\n")
    y = []
    for item in lines:
        arr = item.split(" ")
        if len(arr) != 0 and arr[0] != '':
            mat = np.array([float(arr[-1]), float(arr[-2]), float(arr[-3]), float(arr[-4]), float(arr[-5])])
            y.append(np.mean(mat))
    return y

y1 = read_data('onednn_NN_FP32_6430_UMA.txt')
y2 = read_data('MKL_base_6430.txt')
y3 = read_data('openblas_UMA.txt')
y4 = read_data('GEMMs_LibShalom_FP32_6430.txt')
y5 = read_data('GEMMs_MeGEMM_FP32_6430.txt')

# x 坐标
x = [32 + 32*i for i in range(len(y1))]

# 字体设置
font1 = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 13.5}
font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 15}

# 创建图
fig, axs = plt.subplots(1, 1, figsize=(4.8, 2.7))

# 绘图
l1,=axs.plot(x,y1,color=(46/255,117/255,182/255),marker="D",markersize=3.5)
l2,=axs.plot(x,y2,color=(112/255,173/255,71/255),marker="^",markersize=3.5)
l3,=axs.plot(x,y3,color=(240/255,131/255,58/255),marker="s",markersize=3.5)
l4,=axs.plot(x,y4,color=(165/255,164/255,164/255),marker="*",markersize=3.5)
l5,=axs.plot(x,y5,color=(214/255,39/255,40/255),marker="o",markersize=3.5)

# 设置坐标轴
axs.set_xlabel(r"$M\ $ (NN mode, BF16-FP32)", font1)
axs.set_ylabel("TFLOPS (BF16)", font)
axs.set_xlim(0, 1050)
axs.set_ylim(0, 21.5)
axs.grid(True, c="lightgrey", linestyle='--')
axs.set_xticks([200, 400, 600, 800, 1000])
axs.set_xticklabels(['200', '400', '600', '800', '1000'], fontsize=14)
axs.set_yticks([0, 5, 10, 15, 20])
axs.set_yticklabels(['0', '5', '10', '15', '20'], fontsize=14)
axs.tick_params(direction='in')

# 图例
fig.legend([l1, l2, l3, l4, l5],
           ['oneDNN','MKL','OpenBLAS','LibShalom','TILEFLOW'],
           ncol=3, loc="upper right", borderaxespad=0.0,
           bbox_to_anchor=(0.96, 1.141), columnspacing=0.5,
           frameon=False, prop=font1, handlelength=1.5, handletextpad=0.1)

fig.tight_layout()
plt.subplots_adjust(wspace=0.16, hspace=0.34)

# 展示和保存
plt.show()
fig.savefig('UMA_6430.pdf', bbox_inches='tight')