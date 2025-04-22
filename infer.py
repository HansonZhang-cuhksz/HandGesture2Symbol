import torch
from model import ClassificationModel
import numpy as np

# 初始化模型并加载权重
model = ClassificationModel()
model.load_state_dict(torch.load('classification_model.pth', map_location=torch.device('cpu')))
model.eval()

def predict(input_data):
    assert input_data.shape == (21, 3)
    outputs = model(torch.from_numpy(input_data).unsqueeze(0))
    probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
    predicted_class = int(torch.argmax(outputs, dim=1).item())
    return predicted_class, probabilities