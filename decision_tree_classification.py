import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from read_data import connect4
from data_label_num import data_label_num, combination
from decision_tree import decision_tree


if __name__ == '__main__':
    dataset = connect4(numerical=True, one_hot=False)
    model_data, model_label, train_dataset_num = data_label_num(dataset.train_x, dataset.train_y)
    test_data, test_label, _ = data_label_num(dataset.test_x, dataset.test_y)

    macro_f1_score = []
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
        valid_pred = []
        for i in range(len(valid_dataset[1])):
            res = dt.predict(valid_dataset[0][i])
            valid_pred.append(res)
        score_valid = f1_score(valid_dataset[1], valid_pred, average="macro")
        macro_f1_score.append(score_valid)
        print("验证集：",score_valid)

    # 输出五折交叉验证结果
    print("五折交叉验证结果")
    print("f1_score均值： %.2f" % np.mean(macro_f1_score))
    print("f1_score方差： %.2f" % np.var(macro_f1_score))
    # 全训练集建模
    model_dataset = combination(model_data, model_label)
    dt = decision_tree(max_depth=6)
    dt.train(model_dataset)
    # 测试集评测
    test_dataset = combination(test_data, test_label)
    test_pred = []
    for i in range(len(test_dataset[1])):
        res = dt.predict(test_dataset[0][i])
        test_pred.append(res)
    score_test = f1_score(test_dataset[1], test_pred, average="macro")
    print("测试集：",score_test)