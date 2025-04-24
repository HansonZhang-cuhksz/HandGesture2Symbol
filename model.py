from torch import nn

# # 定义神经网络模型
# class ClassificationModel(nn.Module):
#     def __init__(self, input_size=63, num_classes=12):  # 21*3=63, 12 classes + not_classified
#         super(ClassificationModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, num_classes)
#         self.dropout = nn.Dropout(0.3)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 展平输入 (batch_size, 21*3)
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
        
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
        
#         x = self.fc3(x)
#         return x
    
# 定义改进的神经网络模型
class ClassificationModel(nn.Module):
    def __init__(self, input_size=42, num_classes=12):  # 21*2=42, 12 classes + not_classified
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)  # 增加Dropout比率
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入 (batch_size, 21*2)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x