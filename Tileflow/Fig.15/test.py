import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean


#plt.rcParams['font.sans-serif'] = ['SimSun']  # Windows 系统默认宋体
#plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# ============================================
 
f1 = open('oneDNN_NN.txt', 'r')
xy1=f1.read().split("\n")

f2 = open('MKL_NN.txt', 'r')
xy2=f2.read().split("\n")

f3 = open('OpenBLAS_NN.txt', 'r')
xy3=f3.read().split("\n")

f4 = open('MeGEMM_NN.txt', 'r')
xy4=f4.read().split("\n")


y1=[]
y2=[]
y3=[]
y4=[]
x=[]

for item in xy1:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y1.append(np.mean(mat))


for item in xy2:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y2.append(np.mean(mat))


for item in xy3:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y3.append(np.mean(mat))


for item in xy4:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y4.append(np.mean(mat))

y1 = [v / y1[0] for v in y1]
y2 = [v / y2[0] for v in y2]
y3 = [v / y3[0] for v in y3]
y4 = [v / y4[0] for v in y4]

fig, axs= plt.subplots(1, 2,figsize=(7.5,2.6))

x=[1,2,4,8,16,32]


font1 = {'family' : 'DejaVu Sans',
'weight' : 'normal',
'size' : 13.5,
}

font = {'family' : 'DejaVu Sans',
'weight' : 'normal',
'size' : 15,
}


l1,=axs[0].plot(x,y1,color=(30/255,118/255,179/255),marker="D",markersize=3.5)
l2,=axs[0].plot(x,y2,color=(66/255,165/255,48/255),marker="^",markersize=3.5)
l3,=axs[0].plot(x,y3,color=(240/255,131/255,58/255),marker="s",markersize=3.5)
l4,=axs[0].plot(x,y4,color=(214/255,39/255,40/255),marker="o",markersize=3.5)


axs[0].set_xlabel("Number of threads (NN mode)",font1)
axs[0].set_ylabel("Speedup",font)
axs[0].set_xlim(0,33)
axs[0].grid(True,c="lightgrey",linestyle='--')
axs[0].set_xticks((0,8,16,24,32),('0','8','16','24','32'),fontsize=14)
axs[0].set_yticks((5,10,15,20,25,30),('5','10','15','20','25','30'),fontsize=14)
axs[0].set_ylim(0,32)
axs[0].tick_params(direction='in')

plt.setp(axs[0].get_xticklabels(),fontsize=15)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=15)



f1 = open('oneDNN_NT.txt', 'r')
xy1=f1.read().split("\n")

f2 = open('MKL_NT.txt', 'r')
xy2=f2.read().split("\n")

f3 = open('OpenBLAS_NT.txt', 'r')
xy3=f3.read().split("\n")

f4 = open('MeGEMM_NT.txt', 'r')
xy4=f4.read().split("\n")


y1=[]
y2=[]
y3=[]
y4=[]

for item in xy1:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y1.append(np.mean(mat))


for item in xy2:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y2.append(np.mean(mat))


for item in xy3:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y3.append(np.mean(mat))


for item in xy4:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y4.append(np.mean(mat))

y1 = [v / y1[0] for v in y1]
y2 = [v / y2[0] for v in y2]
y3 = [v / y3[0] for v in y3]
y4 = [v / y4[0] for v in y4]

l1,=axs[1].plot(x,y1,color=(30/255,118/255,179/255),marker="D",markersize=3.5)
l2,=axs[1].plot(x,y2,color=(66/255,165/255,48/255),marker="^",markersize=3.5)
l3,=axs[1].plot(x,y3,color=(240/255,131/255,58/255),marker="s",markersize=3.5)
l4,=axs[1].plot(x,y4,color=(214/255,39/255,40/255),marker="o",markersize=3.5)


axs[1].set_xlabel("Number of threads (NT mode)",font1)
axs[1].set_xlim(0,33)
axs[1].grid(True,c="lightgrey",linestyle='--')
axs[1].set_xticks((0,8,16,24,32),('0','8','16','24','32'),fontsize=14)
axs[1].set_yticks((5,10,15,20),('5','10','15','20'),fontsize=14)
axs[1].set_ylim(0,20.5)
axs[1].tick_params(direction='in')

plt.setp(axs[1].get_xticklabels(),fontsize=15)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=15)


fig.legend([l1,l2,l3,l4], ['oneDNN','MKL','OpenBLAS','TILEFLOW'], ncol=4,
	loc="upper right" , borderaxespad=0.0,bbox_to_anchor=(0.68,1.045), columnspacing = 0.5, frameon=False,
	prop =font1,handlelength=1.2,handletextpad=0.1)
	
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace =0.16, hspace = 0.34)
plt.show()
fig.savefig('Sca_M32.pdf', bbox_inches='tight')
