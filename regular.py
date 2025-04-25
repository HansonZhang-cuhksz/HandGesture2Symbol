import numpy as np

GROUPS = [
    [5, 6],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [17, 18],
    [18, 19],
    [19, 20],
]

def predict(input_data):
    thumb = input_data[4][:-1]
    trust = []
    
    #for group in GROUPS:
    #    first_point = input_data[group[0]][:-1]
    #    second_point = input_data[group[1]][:-1]
    #    thumb_to_first = np.linalg.norm(thumb - first_point)
    #    thumb_to_second = np.linalg.norm(thumb - second_point)
    #    first_to_second = np.linalg.norm(first_point - second_point)
    #    dist = thumb_to_first + thumb_to_second
    #    if first_to_second == 0:
    #        trust.append(np.inf)
    #    else:
    #        trust.append(dist / first_to_second)
    # min_index = np.argmin(trust)
    # return min_index

    for point in range(5, 21):
        trust.append(np.linalg.norm(thumb - point))

    min_index = np.argmin(trust)
    return min_index
    
def get_keypoints(pth):
    with open(pth, 'r') as f:
        out = []
        f.readline()    # skip first line
        for line in f:
            out.append([float(num) for num in line.split()][:-1])
        return out
    
if __name__ == '__main__':
    for i in range(12):
        acc = 0
        for j in range(693):
            input_data = get_keypoints(f'dataset\\{i}\\frame_{j:04d}.txt')
            input_data = np.array(input_data, dtype=np.float32)
            predicted_class = predict(input_data)
            if predicted_class == 0:
                acc += 1
        print(f"Class {i} accuracy: {acc / 693:.2%}")
