import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.signal import windows
import pyfftw
from get_context import get_context

# 设置数据集路径
img_dir = r"C:\Users\chenj\Desktop\Matlab_STCv0\data"
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# 初始化
initstate = [160, 64, 75, 95]  # [x, y, width, height]
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
dist = rs**2 + cs**2  # 到中心的距离
conf = np.exp(-0.5 / alpha * np.sqrt(dist))  # 置信图
conf /= np.sum(conf)  # 归一化

# 使用 pyFFTW 进行 FFT
conf_fft = pyfftw.interfaces.numpy_fft.fft2(conf, planner_effort='FFTW_PATIENT')
conff = pyfftw.interfaces.numpy_fft.fft2(conf, planner_effort='FFTW_PATIENT')

# 存储预计算的权重窗口
hamming_window = windows.hamming(sz[0]).reshape(-1, 1) * windows.hann(sz[1])  # Hamming窗口

sigma = np.mean(target_sz)  # 权重函数w_σ的初始σ_1

window = hamming_window * np.exp(-0.5 / (sigma**2) * dist)  # 组合权重窗口
window /= np.sum(window)  # 归一化

# 时空上下文模型初始化
Hstcf = None
maxconf = []

for frame_num, img_file in enumerate(img_files):
    sigma *= scale  # 在Eq.(15)中更新尺度
    window = hamming_window * np.exp(-0.5 / (sigma**2) * dist)  # 更新权重
    window /= np.sum(window)  # 归一化

    # 加载图像
    img = cv2.imread(os.path.join(img_dir, img_file))
    if img.shape[2] > 1:
        im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        im = img

    contextprior = get_context(im, pos, sz, window)  # 获取上下文模型

    if frame_num > 0:
        # 使用 pyFFTW 进行 FFT 和 IFFT
        confmap_fft = pyfftw.interfaces.numpy_fft.fft2(contextprior, planner_effort='FFTW_PATIENT')
        confmap = np.real(pyfftw.interfaces.numpy_fft.ifft2(Hstcf * confmap_fft, planner_effort='FFTW_PATIENT'))
        row, col = np.unravel_index(np.argmax(confmap), confmap.shape)  # 目标位置
        pos = pos - sz / 2 + [row, col] + 1  # 更新位置

        # 重新计算上下文先验和置信图
        contextprior = get_context(im, pos, sz, window)
        conftmp_fft = pyfftw.interfaces.numpy_fft.fft2(contextprior, planner_effort='FFTW_PATIENT')
        conftmp = np.real(pyfftw.interfaces.numpy_fft.ifft2(Hstcf * conftmp_fft, planner_effort='FFTW_PATIENT'))
        maxconf.append(np.max(conftmp))

        # 更新尺度
        if (frame_num % (num + 2)) == 0:
            scale_curr = np.sum([np.sqrt(maxconf[frame_num - kk] / maxconf[frame_num - kk - 1]) for kk in range(1, num + 1)])
            scale = (1 - lambda_) * scale + lambda_ * (scale_curr / num)
    # 更新空间上下文模型
    contextprior = get_context(im, pos, sz, window)
    contextprior_fft = pyfftw.interfaces.numpy_fft.fft2(contextprior, planner_effort='FFTW_PATIENT')
    hscf = conff / (contextprior_fft + np.finfo(float).eps)  # 更新hscf

    # 更新时空上下文模型
    if frame_num == 0:
        Hstcf = hscf
    else:
        Hstcf = (1 - rho) * Hstcf + rho * hscf  # 更新Hstcf

    # 可视化
    target_sz[0] *= scale  # 更新目标大小
    rect_position = [pos[1] - target_sz[1] / 2, pos[0] - target_sz[0] / 2, target_sz[1], target_sz[0]]

    plt.imshow(img, cmap='gray')
    plt.gca().add_patch(plt.Rectangle((rect_position[0], rect_position[1]), rect_position[2], rect_position[3],
                                        linewidth=4, edgecolor='r', facecolor='none'))
    plt.text(5, 18, f'#{frame_num + 1}', color='yellow', fontweight='bold', fontsize=20)
    plt.axis('off')
    plt.pause(0.001)
    plt.clf()  # 清空当前图形

plt.show()  # 显示最终可视化结果