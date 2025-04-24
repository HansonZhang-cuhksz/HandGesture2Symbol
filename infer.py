import torch
from model import ClassificationModel
import numpy as np

# 初始化模型并加载权重
model = ClassificationModel()
model.load_state_dict(torch.load('classification_model2.pth', map_location=torch.device('cpu')))
model.eval()

def get_keypoints(pth):
    with open(pth, 'r') as f:
        out = []
        f.readline()    # skip first line
        for line in f:
            out.append([float(num) for num in line.split()][:-1])
        return out

def predict(input_data):
    assert input_data.shape == (21, 2)
    outputs = model(torch.from_numpy(input_data).unsqueeze(0))
    probabilities = torch.softmax(outputs, dim=1).squeeze().detach().numpy()
    predicted_class = int(torch.argmax(outputs, dim=1).item())
    return predicted_class, probabilities

if __name__ == '__main__':
    # input_data = np.random.rand(21, 3).astype(np.float32)
    input_data = get_keypoints('dataset\\none\\frame_0001.txt')
    input_data = np.array(input_data, dtype=np.float32)
    predicted_class, probabilities = predict(input_data)
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")