from read_data import connect4
import copy

def data_label_num(x, y):
    data = x
    label = []
    for x_num, data_cont in enumerate(x):
        for y_num, cont in enumerate(data_cont):
            if cont == "b":
                data[x_num][y_num] = 0
            elif cont == "x":
                data[x_num][y_num] = 1
            else:
                data[x_num][y_num] = 2
    dataset_num = copy.deepcopy(data)
    for step, label_cont in enumerate(y):
        if label_cont == "loss":
            label.append(0)
            dataset_num[step].append(0)
        elif label_cont == "win":
            label.append(1)
            dataset_num[step].append(1)
        else:
            label.append(2)
            dataset_num[step].append(2)
    return data, label, dataset_num

if __name__ == '__main__':
    dataset = connect4(numerical=True, one_hot=False)
    data, label, dataset_num = data_label_num(dataset.train_x, dataset.train_y)