import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean


#plt.rcParams['font.sans-serif'] = ['SimSun']  # Windows 系统默认宋体
#plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# ============================================
 
f1 = open('onednn_NN_BF16.txt', 'r')
xy1=f1.read().split("\n")

f4 = open('MeGEMM_NN_BF16.txt', 'r')
xy4=f4.read().split("\n")


y1=[]
y4=[]
x=[]

for item in xy1:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y1.append(np.mean(mat))


for item in xy4:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y4.append(np.mean(mat))


fig, axs= plt.subplots(1, 4,figsize=(15.2,2.48))

x = []
x1=[]
for i in range(len(y1)):
	x.append(32+32*i)


#temp1 = [a / b for a, b in zip(y3, y1)]
#temp2 = [a / b for a, b in zip(y3, y2)]


font1 = {'family' : 'DejaVu Sans',
'weight' : 'normal',
'size' : 13.5,
}

font = {'family' : 'DejaVu Sans',
'weight' : 'normal',
'size' : 15,
}


l1,=axs[0].plot(x,y1,color=(46/255,117/255,182/255),marker="D",markersize=3.5)
l4,=axs[0].plot(x,y4,color=(214/255,39/255,40/255),marker="o",markersize=3.5)


axs[0].set_xlabel(r"$M\ $ (NN mode, BF16-BF16)",font1)
axs[0].set_ylabel("TFLOPS (BF16)",font)
axs[0].set_xlim(0,1050)
axs[0].grid(True,c="lightgrey",linestyle='--')
axs[0].set_xticks((200,400,600,800,1000),('200','400','600','800','1000'),fontsize=14)
axs[0].set_yticks((0, 10,20,30,40,45),('0', '10','20','30','40','45'),fontsize=14)
axs[0].set_ylim(0,47.5)
axs[0].tick_params(direction='in')

plt.setp(axs[0].get_xticklabels(),fontsize=15)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=15)

speedup_1 = gmean([a/b for a,b in zip(y4,y1)])
print("Subplot1 MeGEMM vs oneDNN Speedup:", speedup_1)


y1=[]
y2=[]
y3=[]
y4=[]

f1 = open('onednn_NN_FP32.txt', 'r')
xy1=f1.read().split("\n")

f2 = open('MKL_NN_FP32.txt', 'r')
xy2=f2.read().split("\n")

f3 = open('OpenBLAS_NN_FP32.txt', 'r')
xy3=f3.read().split("\n")

f4 = open('MeGEMM_NN_FP32.txt', 'r')
xy4=f4.read().split("\n")


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


l1,=axs[1].plot(x,y1,color=(46/255,117/255,182/255),marker="D",markersize=3.5)
l2,=axs[1].plot(x,y2,color=(112/255,173/255,71/255),marker="^",markersize=3.5)
l3,=axs[1].plot(x,y3,color=(240/255,131/255,58/255),marker="s",markersize=3.5)
l4,=axs[1].plot(x,y4,color=(214/255,39/255,40/255),marker="o",markersize=3.5)

axs[1].set_xlabel(r"$M\ $ (NN mode, BF16-FP32)",font1)
axs[1].set_xlim(0,1050)
axs[1].grid(True,c="lightgrey",linestyle='--')
axs[1].set_xticks((200,400,600,800,1000),('200','400','600','800','1000'),fontsize=14)
axs[1].set_yticks((0, 10,20,30,40,45),('0', '10','20','30','40','45'),fontsize=14)
axs[1].set_ylim(0,47.5)
plt.setp(axs[1].get_xticklabels(),fontsize=15)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=15)
axs[1].tick_params(direction='in')

speedup_21 = gmean([a/b for a,b in zip(y4,y1)])
speedup_22 = gmean([a/b for a,b in zip(y4,y2)])
speedup_23 = gmean([a/b for a,b in zip(y4,y3)])

print("Subplot2 MeGEMM vs oneDNN Speedup:", speedup_21)
print("Subplot2 MeGEMM vs MKL Speedup:", speedup_22)
print("Subplot2 MeGEMM vs OpenBLAS Speedup:", speedup_23)



f1 = open('onednn_NT_BF16.txt', 'r')
xy1=f1.read().split("\n")

f4 = open('MeGEMM_NT_BF16.txt', 'r')
xy4=f4.read().split("\n")


y1=[]
y4=[]

for item in xy1:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y1.append(np.mean(mat))


for item in xy4:
	arr = item.split(" ")
	if(len(arr) != 0 and arr[0]!= ''):
		mat = np.array([float(arr[-1]),float(arr[-2]),float(arr[-3]),float(arr[-4]), float(arr[-5])])
		y4.append(np.mean(mat))


l1,=axs[2].plot(x,y1,color=(46/255,117/255,182/255),marker="D",markersize=3.5)
l4,=axs[2].plot(x,y4,color=(214/255,39/255,40/255),marker="o",markersize=3.5)


axs[2].set_xlabel(r"$M\ $ (NT mode, BF16-BF16)",font1)
axs[2].set_xlim(0,1050)
axs[2].grid(True,c="lightgrey",linestyle='--')
axs[2].set_xticks((200,400,600,800,1000),('200','400','600','800','1000'),fontsize=14)
axs[2].set_yticks((0, 10,20,30,40,45),('0', '10','20','30','40','45'),fontsize=14)
axs[2].set_ylim(0,47.5)
axs[2].tick_params(direction='in')

plt.setp(axs[2].get_xticklabels(),fontsize=15)
axs[2].set_yticklabels(axs[2].get_yticks(), fontsize=15)


speedup_3 = gmean([a/b for a,b in zip(y4,y1)])
print("Subplot3 MeGEMM vs oneDNN Speedup:", speedup_3)




y1=[]
y2=[]
y3=[]
y4=[]

f1 = open('onednn_NT_FP32.txt', 'r')
xy1=f1.read().split("\n")

f2 = open('MKL_NT_FP32.txt', 'r')
xy2=f2.read().split("\n")

f3 = open('OpenBLAS_NT_FP32.txt', 'r')
xy3=f3.read().split("\n")

f4 = open('MeGEMM_NT_FP32.txt', 'r')
xy4=f4.read().split("\n")


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


l1,=axs[3].plot(x,y1,color=(46/255,117/255,182/255),marker="D",markersize=3.5)
l2,=axs[3].plot(x,y2,color=(112/255,173/255,71/255),marker="^",markersize=3.5)
l3,=axs[3].plot(x,y3,color=(240/255,131/255,58/255),marker="s",markersize=3.5)
l4,=axs[3].plot(x,y4,color=(214/255,39/255,40/255),marker="o",markersize=3.5)

axs[3].set_xlabel(r"$M\ $ (NT mode, BF16-FP32)",font1)
axs[3].set_xlim(0,1050)
axs[3].grid(True,c="lightgrey",linestyle='--')
axs[3].set_xticks((200,400,600,800,1000),('200','400','600','800','1000'),fontsize=14)
axs[3].set_yticks((0, 10,20,30,40,45),('0', '10','20','30','40','45'),fontsize=14)
axs[3].set_ylim(0,47.5)
plt.setp(axs[3].get_xticklabels(),fontsize=15)
axs[3].set_yticklabels(axs[3].get_yticks(), fontsize=15)
axs[3].tick_params(direction='in')

speedup_41 = gmean([a/b for a,b in zip(y4,y1)])
speedup_42 = gmean([a/b for a,b in zip(y4,y2)])
speedup_43 = gmean([a/b for a,b in zip(y4,y3)])

print("Subplot4 MeGEMM vs oneDNN Speedup:", speedup_41)
print("Subplot4 MeGEMM vs MKL Speedup:", speedup_42)
print("Subplot4 MeGEMM vs OpenBLAS Speedup:", speedup_43)

"""
fig.legend([l1,l2,l3,l4], ['oneDNN','MKL','OpenBLAS','MeGEMM'], ncol=4,
	loc="upper right" , borderaxespad=0.0,bbox_to_anchor=(0.35,1.03), columnspacing = 0.5, frameon=False,
	prop =font1,handlelength=1.5,handletextpad=0.1)
"""

print("vs oneDNN:",gmean([speedup_1,speedup_21,speedup_3,speedup_41]))
print("vs MKL:",gmean([speedup_22,speedup_42]))
print("vs OpenBLAS:",gmean([speedup_23,speedup_43]))


fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace =0.16, hspace = 0.34)
plt.show()
fig.savefig('GEMM_6448H.pdf', bbox_inches='tight')
