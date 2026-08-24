import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

font = {'family':'DejaVu Sans','weight':'normal','size':13.5}

def read_perf(file):
    y=[]
    with open(file,'r') as f:
        for line in f:
            arr=line.split()
            if len(arr)>0:
                mat=np.array([ float(arr[-2]) / (float(arr[-1])+float(arr[-2]))*100])
                y.append(np.mean(mat))
    return np.array(y)

files1=[
'oneDNN.txt',
'MKL.txt',
'OpenBLAS.txt',
'MeGEMM_interleave.txt',
'MeGEMM_aware.txt'
]

data1=[read_perf(f) for f in files1]

files2=[
'../N=25600/oneDNN.txt',
'../N=25600/MKL.txt',
'../N=25600/OpenBLAS.txt',
'../N=25600/MeGEMM_interleave.txt',
'../N=25600/MeGEMM_aware.txt'
]

data2=[read_perf(f) for f in files2]


labels=('oneDNN','MKL','OpenBLAS','TILEFLOW_interleave','TILEFLOW_aware')

"""
colors = [
(0.6, 0.6, 0.6),
(66/255,165/255,48/255),
(240/255,131/255,58/255),
(214/255,39/255,40/255),
(214/255,39/255,40/255)
]
"""
colors = [
    '#CCCCCC',              # gray   - base
    '#4C72B0',              # blue   - +f
    '#55A868',              # green  - +interleave  
    '#C44E52',              # red    - +aware
    '#C44E52'              # red    - +aware
]

x=np.arange(len(data1[0]))*3.6
width=0.6

mpl.rcParams['hatch.linewidth'] = 2.0

fig,axs=plt.subplots(1,2,figsize=(9.,2.5))

# -------- 第一个子图 --------
ax=axs[0]

offset=(len(data1)-1)/2

for j,y in enumerate(data1):

    xpos = x+(j-offset)*width

    if labels[j] == 'MeGEMM_aware':

        rect = ax.bar(
            xpos, y, width,
            facecolor='none',
            edgecolor='black',
            hatch='////', zorder=3,
            linewidth=1.2
        )

        for r in rect:
            r._hatch_color = (214/255,39/255,40/255)

    else:
        ax.bar(
            xpos, y, width,
            color=colors[j], zorder=3,
            edgecolor='black'
        )
ax.tick_params(axis='x', which='both', length=0)
ax.set_xticks(x)
ax.set_xticklabels(("16","32","64","128","256"), fontsize=15)

ax.set_yticks((0,10,20,30,40,50))
ax.set_yticklabels(ax.get_yticks(), fontsize=15)

ax.set_xlim(-1.8,16.2)
ax.set_ylim(0,52)

ax.set_xlabel(r"$M\ $ ($N$=28672, $K$=8192)",font)
ax.set_ylabel('% of overall',font)

ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

# -------- 第二个子图空着 --------
ax=axs[1]

offset=(len(data2)-1)/2

for j,y in enumerate(data2):

    xpos = x+(j-offset)*width

    if labels[j] == 'TILEFLOW_aware':

        rect = ax.bar(
            xpos, y, width,
            facecolor='none',
            edgecolor='black',
            hatch='////', zorder=3,
            linewidth=1.2
        )

        for r in rect:
            r._hatch_color = (214/255,39/255,40/255)

    else:
        ax.bar(
            xpos, y, width,
            color=colors[j], zorder=3,
            edgecolor='black'
        )

ax.set_xticks(x)
ax.set_xticklabels(("16","32","64","128","256"), fontsize=15)

ax.set_yticks((0,10,20,30,40,50))
ax.set_yticklabels(ax.get_yticks(), fontsize=15)

ax.set_xlim(-1.8,16.2)
ax.set_ylim(0,52)

ax.set_xlabel(r"$M\ $ ($N$=25600, $K$=5120)",font)

ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

# -------- legend --------
aware_handle = Rectangle(
    (0,0),1,1,
    facecolor='none',
    edgecolor='black',
    hatch='////',
    linewidth=1.2
)

aware_handle._hatch_color = (214/255,39/255,40/255)

legend_handles = [
    Rectangle((0,0),1,1,facecolor=colors[0],edgecolor='black'),
    Rectangle((0,0),1,1,facecolor=colors[1],edgecolor='black'),
    Rectangle((0,0),1,1,facecolor=colors[2],edgecolor='black'),
    Rectangle((0,0),1,1,facecolor=(214/255,39/255,40/255),edgecolor='black'),
    aware_handle
]

fig.legend(
    legend_handles, labels,
    loc="upper right",
    bbox_to_anchor=(0.9,1.1),
    ncol=5,
    columnspacing=0.6,
    frameon=False,
    prop=font,
    handlelength=0.8,
    handletextpad=0.2
)

ax.tick_params(axis='x', which='both', length=0)

fig.tight_layout()
plt.savefig('Traffic.pdf',bbox_inches='tight')
plt.show()