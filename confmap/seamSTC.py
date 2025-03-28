import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.signal import windows
import pyfftw
from get_context import get_context
import random

# 设置数据集路径

img1_path = r'./62.jpg'
img2_path = r'./150.jpg'

# img_label = r'./1B882_3061.txt'

# with open(img_label, 'r') as file:
#     coordinates = file.readlines()
#     coordinates = [list(map(float, line.strip().split())) for line in coordinates]

# x, y = coordinates[0][0]-75, coordinates[1][0]-75

# print(f"Coordinates: {coordinates}")
x = 160-75
y = 140-75
print(f"x: {x}, y: {y}")

# 初始化
initstate = [x, y, 150, 150]  # [x, y, width, height]
pos = [initstate[1] + initstate[3] / 2, initstate[0] + initstate[2] / 2]  # 目标中心
target_sz = [initstate[3], initstate[2]]  # [height, width]
padding = 1  # 目标外围的额外区域
rho = 0.075
sz = np.floor(np.array(target_sz) * (1 + padding)).astype(int)  # 上下文区域大小
print(sz)

# 尺度更新参数
scale = 1  # 初始尺度比率
lambda_ = 0.25  # λ in Eq.(15)
num = 5  # 平均帧数

# 存储预计算的置信图参数
alpha = 2.25  # α in Eq.(6)
cs, rs = np.meshgrid(np.arange(1, sz[1] + 1) - sz[1] // 2, np.arange(1, sz[0] + 1) - sz[0] // 2)
dist = rs**2 + cs**2  # 到中心的距禂
conf = np.exp(-0.5 / alpha * np.sqrt(dist))  # 置信图
conf /= np.sum(conf)  # 归一化

# 使用 pyFFTW 进行 FFT
conff = pyfftw.interfaces.numpy_fft.fft2(conf, planner_effort='FFTW_PATIENT')

# 存储预计算的权重窗口
hamming_window = windows.hamming(sz[0]).reshape(-1, 1) * windows.hann(sz[1])  # Hamming窗口

sigma = np.mean(target_sz)  # 权重函数w_σ的初始σ_1

window = hamming_window * np.exp(-0.5 / (sigma**2) * dist)  # 组合权重窗口
window /= np.sum(window)  # 归一化

# 加载图像
img = cv2.imread(img1_path)

if img.shape[2] > 1:
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    im = img

contextprior = get_context(im, pos, sz, window)  # 获取上下文模型
contextprior_fft = pyfftw.interfaces.numpy_fft.fft2(contextprior, planner_effort='FFTW_PATIENT')
hscf = conff / (contextprior_fft + np.finfo(float).eps)  # 更新hscf

# 加载图像
img = cv2.imread(img2_path)

if img.shape[2] > 1:
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    im = img
contextprior = get_context(im, pos, sz, window)  # 获取上下文模型

confmap_fft = pyfftw.interfaces.numpy_fft.fft2(contextprior, planner_effort='FFTW_PATIENT')
confmap = np.real(pyfftw.interfaces.numpy_fft.ifft2(hscf * confmap_fft, planner_effort='FFTW_PATIENT'))



# 可视化
target_sz[0] *= scale  # 更新目标大小
rect_position = [pos[1] - target_sz[1] / 2, pos[0] - target_sz[0] / 2, target_sz[1], target_sz[0]]

# 放大置信图
scale_factor = 2
conf_resized = cv2.resize(confmap, (confmap.shape[1] * scale_factor, confmap.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)

# 绘制三维置信图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(conf_resized.shape[1]), np.arange(conf_resized.shape[0]))
ax.plot_surface(X, Y, conf_resized, cmap='hot')
ax.set_title('3D Confidence Map (Resized)')
plt.show()

plt.imshow(img, cmap='gray')
plt.gca().add_patch(plt.Rectangle((rect_position[0], rect_position[1]), rect_position[2], rect_position[3],
                                    linewidth=4, edgecolor='r', facecolor='none'))
plt.axis('off')
plt.pause(10)
plt.clf()  # 清空当前图形