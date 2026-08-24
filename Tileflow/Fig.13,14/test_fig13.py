import numpy as np
import matplotlib.pyplot as plt

def load_txt(file_path):
    with open(file_path, "r") as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

# 数据
# baseline: IPEX+oneDNN
ipex_onednn = load_txt("NT.txt")

# 其他两组
ipex = load_txt("prepack.txt")
ipex_megemm = load_txt("prepack_ours.txt")
ipex_megemm_no_prepack = load_txt("NT_ours.txt")

# speedup，相对于 IPEX+oneDNN
speedup_onednn = np.ones_like(ipex_onednn)
#speedup_ipex = ipex / ipex_onednn
speedup_megemm = ipex_megemm / ipex
speedup_no_prepack = ipex_megemm_no_prepack/ ipex_onednn

# x 轴
x = np.arange(len(ipex_onednn))
width = 0.25

# 字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 颜色：左蓝，中绿，右红
colors = ['#4C72B0', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(7.5, 2.5))

font1 = {'family' : 'DejaVu Sans',
'weight' : 'normal',
'size' : 14.5,
}

font = {'family' : 'DejaVu Sans',
'weight' : 'normal',
'size' : 14,
}

#l1=plt.bar(x - width, speedup_onednn, width, color=colors[0], edgecolor='black')
#l2=plt.bar(x,         speedup_ipex,   width, color=colors[1], edgecolor='black')
l2=plt.bar(x -0.5* width, speedup_megemm, width, color=colors[1], zorder=3, edgecolor='black')
l3=plt.bar(x +0.5* width, speedup_no_prepack, width, color=colors[2], zorder=3, edgecolor='black')



plt.xlabel('Batch size', font1)
plt.ylabel('speedup', font1)
#plt.xticks(x, x + 1)

ax.set_xlim(-0.5, 15.5, font1)   # 或 (0, 33)
ax.set_yticks((0,0.5,1.0,1.5,2.0))
ax.set_ylim(0, 2.05, font1)
ax.set_yticklabels(ax.get_yticks(), fontsize=13.5)
ax.set_xticks(x)
ax.set_xticklabels(("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"), 
        fontsize=13.5)

fig.legend([l2,l3], ['AMX-friendly layout','general-purpose layout'], ncol=2,
    loc="upper right" , borderaxespad=0.0,bbox_to_anchor=(0.735,1.045),
    columnspacing = 0.5, frameon=False,
    prop =font,handlelength=0.8,handletextpad=0.1)


speedup_megemm = ipex_megemm / ipex
speedup_no_prepack = ipex_megemm_no_prepack / ipex_onednn

print("AMX layout:")
print("  Mean =", np.mean(speedup_megemm))
print("  Max  =", np.max(speedup_megemm))

print("NT mode:")
print("  Mean =", np.mean(speedup_no_prepack))
print("  Max  =", np.max(speedup_no_prepack))

ax.tick_params(axis='x', which='both', length=0)
ax.grid(axis='y', color='lightgrey', linestyle='--', alpha=0.8)

plt.tight_layout()
plt.savefig('e2e.pdf',bbox_inches='tight')
plt.show()