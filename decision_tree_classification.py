from decision_tree import decision_tree
from read_data import connect4
from sklearn.model_selection import KFold
from data_label_num import data_label_num, combination
import pandas as pd

if __name__ == '__main__':
    dataset = connect4(numerical=True, one_hot=False)
    _, _, train_dataset_num = data_label_num(dataset.train_x, dataset.train_y)
    test_data, test_label, _ = data_label_num(dataset.test_x, dataset.test_y)

    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    for train_index, valid_index in kf.split(train_dataset_num):
        train_data = pd.DataFrame(train_dataset_num.copy()).drop(valid_index)
        valid_data = pd.DataFrame(train_dataset_num.copy()).drop(train_index)

        train_data_y = train_data.iloc[:, -1]
        valid_data_y = valid_data.iloc[:, -1]
        train_data_x = train_data.drop(axis=1, columns=42)
        valid_data_x = valid_data.drop(axis=1, columns=42)

        train_dataset = combination(train_data_x.values.tolist(), train_data_y.values.tolist())
        dt = decision_tree(max_depth=6)
        dt.train(train_dataset)
        valid_dataset = combination(valid_data_x.values.tolist(), valid_data_y.values.tolist())
        valid_right = 0
        for i in range(len(valid_dataset[1])):
            res = dt.predict(valid_dataset[0][i])
            if res == valid_dataset[1][i]:
                valid_right += 1
        print("验证集：",valid_right / len(valid_dataset[1]))

    test_dataset = combination(test_data, test_label)
    test_right = 0
    for i in range(len(test_dataset[1])):
        res = dt.predict(test_dataset[0][i])
        if res == test_dataset[1][i]:
            test_right += 1
    print("测试集：",test_right / len(test_dataset[1]))