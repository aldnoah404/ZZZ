import numpy as np  

def get_context(im, pos, sz, window):  
    # 获取并处理上下文区域  
    xs = np.floor(pos[1] + (np.arange(sz[1])) - (sz[1] / 2) ).astype(int)  
    ys = np.floor(pos[0] + (np.arange(sz[0])) - (sz[0] / 2) ).astype(int)  

    # 检查是否超出边界，并将其设置为边界处的值  
    xs[xs < 0] = 0  
    ys[ys < 0] = 0  
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    # Extract image in context region  
    out = im[ys[:, None], xs]  # 使用广播提取图像区域  

    # 预处理窗口  
    out = out.astype(np.float64)  # 将图像转换为双精度浮点数  
    out = (out - np.mean(out))  # 归一化  
    out = window * out  # 使用窗口加权强度作为上下文先验模型  

    return out 