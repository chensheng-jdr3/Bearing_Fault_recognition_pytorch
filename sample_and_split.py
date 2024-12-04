import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import train_test_split

def sample(path, label, numbers=1000):
    files = os.listdir(path)
    X = np.arange(512)
    for file in files:
        data = sio.loadmat(os.path.join(path, file))
        name = file[:-4]
        if len(name) > 2:
            head = 'X' + name + '_DE_time'
        else:
            head = 'X0' + name + '_DE_time'
        data = data[head].reshape(-1)
        stride = int((len(data) - 512) / (numbers - 1))
        i = 0
        while i < len(data):
            j = i + 512
            if j > len(data):
                break
            x = data[i:j]
            X = np.row_stack([X, x])
            i = i + stride
    X = np.delete(X, 0, axis=0)
    y = np.empty(len(X))
    y.fill(label)
    return X, y


if __name__ == '__main__':
    # normal:4000, 1000/file, label:0
    # inner: 4000, 250/file, label:1
    # roll:4000, 250/file, label:2
    # outer:4000, 142/file, label:3
    path_normal = r'raw-data\Normal Baseline Data'
    path_inner = r'raw-data\12k Drive End Bearing Fault Data\内圈故障'
    path_roll = r'raw-data\12k Drive End Bearing Fault Data\滚动体故障'
    path_outer = r'raw-data\12k Drive End Bearing Fault Data\外圈故障'
    
    # 采样不同类数据并贴上标签
    x_noraml, y_normal = sample(path_normal, label=0)
    x_inner, y_inner = sample(path_inner, label=1, numbers=250)
    x_roll, y_roll = sample(path_roll, label=2, numbers=250)
    x_outer, y_outer = sample(path_outer, label=3, numbers=143)
    print(x_noraml.shape, y_normal.shape)
    print(x_inner.shape, y_inner.shape)
    print(x_roll.shape, y_roll.shape)
    print(x_outer.shape, y_outer.shape)

    # 把不同类数据合并
    x = x_noraml
    x = np.row_stack([x, x_inner])
    x = np.row_stack([x, x_roll])
    x = np.row_stack([x, x_outer])
    os.makedirs(os.path.dirname(r'Data\x'), exist_ok=True)
    np.savetxt(r'Data\x', x)

    # 把不同类标签合并
    y = np.append(y_normal, y_inner)
    y = np.append(y, y_roll)
    y = np.append(y, y_outer)
    os.makedirs(os.path.dirname(r'Data\y'), exist_ok=True)
    np.savetxt(r'Data\y', y)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=66)
    os.makedirs(os.path.dirname(r'Data\x_train'), exist_ok=True)
    os.makedirs(os.path.dirname(r'Data\y_train'), exist_ok=True)
    os.makedirs(os.path.dirname(r'Data\x_test'), exist_ok=True)
    os.makedirs(os.path.dirname(r'Data\y_test'), exist_ok=True)
    np.savetxt(r'Data\x_train', x_train)
    np.savetxt(r'Data\y_train', y_train)
    np.savetxt(r'Data\x_test', x_test)
    np.savetxt(r'Data\y_test', y_test)

    # 训练集归一化
    # 之所以要归一化是因为神经网络对输入数据的幅值敏感
    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    x_train_std = (x_train - x_train_min) / (x_train_max - x_train_min)
    x_train_std = x_train_std.astype(np.float32)
    os.makedirs(os.path.dirname(r'Data\x_train_std'), exist_ok=True)
    np.savetxt(r'Data\x_train_std', x_train_std)
    # 测试集归一化
    # 测试集的归一化要用训练集的最大最小值
    x_test_std = (x_test - x_train_min) / (x_train_max - x_train_min)
    x_test_std = x_test_std.astype(np.float32)
    os.makedirs(os.path.dirname(r'Data\x_test_std'), exist_ok=True)
    np.savetxt(r'Data\x_test_std', x_test_std)
