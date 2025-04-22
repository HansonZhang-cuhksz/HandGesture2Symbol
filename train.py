import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from model import ClassificationModel

def get_keypoints(pth):
    with open(pth, 'r') as f:
        out = []
        f.readline()    # skip first line
        for line in f:
            out.append([float(num) for num in line.split()])
        return out

# 定义自定义数据集类
class VectorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'classification_model.pth')
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.show()
    
    return model

def load_data():
    X = []
    y = []
    import os
    for i in list(range(12)) + ['none']:
        files = os.listdir(os.path.join('dataset', str(i)))
        for file in files:
            keypoints = get_keypoints(os.path.join('dataset', str(i), file))
            X.append(keypoints)
            y.append(int(i) if i != 'none' else 12)
    return np.array(X), np.array(y)
            

if __name__ == '__main__':
    # 1. 准备数据 - 这里使用随机数据作为示例
    # 实际应用中应该替换为你的真实数据
    # num_samples = 1000
    # X = np.random.rand(num_samples, 21, 3)  # 1000个样本，每个是21x3的向量
    # y = np.random.randint(0, 13, size=num_samples)  # 0-12的类别标签
    X, y = load_data()

    # 检查数据形状
    print(f"Data shape: {X.shape}")  # 应该是 (n_samples, 21, 3)
    print(f"Labels shape: {y.shape}")  # 应该是 (n_samples,)
    print(f"Label distribution: {np.bincount(y)}")  # 查看各类别样本分布
    
    # 划分训练集和测试集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = VectorDataset(X_train, y_train)
    val_dataset = VectorDataset(X_val, y_val)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # 2. 初始化模型
    model = ClassificationModel()
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练模型
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10000)
    
    # 5. 评估模型
    trained_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 打印分类报告
    class_names = [str(i) for i in range(12)] + ['not_classified']
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # # 6. 保存模型
    # torch.save(trained_model.state_dict(), 'classification_model.pth')