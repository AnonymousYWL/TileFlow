import numpy as np
import matplotlib.pyplot as plt

font = {'family':'DejaVu Sans','weight':'normal','size':13.5}

def read_perf(file):
    y=[]
    with open(file,'r') as f:
        for line in f:
            arr=line.split()
            if len(arr)>0:
                mat=np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]),float(arr[-5])])
                y.append(np.mean(mat))
    return np.array(y)

# 两组实验文件
files1=[
'NN_base.txt',
'NN+f+c.txt',
'NN_interleave.txt',
'NN_aware.txt'
]

files2=[
'../N=25600/NN_base.txt',
'../N=25600/NN+f+c.txt',
'../N=25600/NN_interleave.txt',
'../N=25600/NN_aware.txt'
]

data1=[read_perf(f) for f in files1]
data2=[read_perf(f) for f in files2]

data1_step = [np.ones_like(data1[0])] + [data1[i]/data1[i-1] for i in range(1,4)]
data2_step = [np.ones_like(data2[0])] + [data2[i]/data2[i-1] for i in range(1,4)]
print([np.mean(d) for d in data1_step])
print([np.mean(d) for d in data2_step])
print([np.mean((data1_step[i]+data2_step[i])/2) for i in range(4)])



def to_speedup(data):
    base=data[0].copy()
    return [base/base, data[1]/base, data[2]/base, data[3]/base]

data1=to_speedup(data1)
data2=to_speedup(data2)

labels=('base','+f','+interleave','+aware')
"""
colors = [
(0.6, 0.6, 0.6),                # gray   base / oneDNN (baseline)
(66/255,165/255,48/255),        # green  v1 / MKL
(240/255,131/255,58/255),       # orange v2 / OpenBLAS
(214/255,39/255,40/255)         # red    MeGEMM
]"""
colors = [
    '#CCCCCC',              # gray   - base
    '#4C72B0',              # blue   - +f
    '#55A868',              # green  - +interleave  
    '#C44E52',              # red    - +aware
]
x=np.arange(len(data1[0]))*3
width=0.6

fig,axs=plt.subplots(1,2,figsize=(9.,2.5))

for i,(ax,data) in enumerate(zip(axs,[data1,data2])):

    bars=[]
    for j,y in enumerate(data):
        bars.append(ax.bar(x+(j-1.5)*width,y,width,
                           color=colors[j],edgecolor='black'))

    ax.set_xticks(x)
    ax.set_xticklabels(("16","32","64","128","256"), fontsize=14)

    ax.tick_params(axis='x', which='both', length=0)

    if i==0:
        ax.set_ylim(0,2.8)
        ax.set_yticks((0,0.5,1.0,1.5,2.0,2.5))
        ax.set_xlabel(r"$M\ $ ($N=28672$, $K=8192$)",font)
    else:
        ax.set_ylim(0,2.1)
        ax.set_yticks((0,0.5,1.0,1.5,2.0))
        ax.set_xlabel(r"$M\ $ ($N=25600$, $K=5120$)",font)

    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    ax.set_xlim(-1.4,13.4)
    
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

axs[0].set_ylabel('Speedup',font)

fig.legend([b[0] for b in bars],labels,
           loc="upper right",
           bbox_to_anchor=(0.51,1.09),
           ncol=4,
           columnspacing=0.6,
           frameon=False,
           prop=font,
           handlelength=0.8,
           handletextpad=0.2)

fig.tight_layout()
plt.show()
fig.savefig('Ablation.pdf',bbox_inches='tight')