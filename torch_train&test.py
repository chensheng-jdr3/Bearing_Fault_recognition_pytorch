import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from ResNet_torch import create_ResNet

# 如果有GPU可用，使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
x_train = np.loadtxt(r'Data/x_train').reshape(-1, 1, 512).astype(np.float32)
y_train = np.loadtxt(r'Data/y_train').astype(np.int64)
x_test = np.loadtxt(r'Data/x_test').reshape(-1, 1, 512).astype(np.float32)
y_test = np.loadtxt(r'Data/y_test').astype(np.int64)

# 转换为Dataset
train_db = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
test_db = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

# 转换为DataLoader
train_loader = DataLoader(train_db, batch_size=512, shuffle=True)
test_loader = DataLoader(test_db, batch_size=512, shuffle=True)

# 定义模型
model = create_ResNet()
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(20):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct = torch.eq(pred, y).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
        print('epoch:', epoch, 'acc:', acc)

# 保存模型
torch.save(model, 'model.pth')