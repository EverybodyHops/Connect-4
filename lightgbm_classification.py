import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from read_data import connect4
from data_label_num import data_label_num
from lightgbm_model import train_model, eval_model

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

        gbm_model, evals_result = train_model(train_data_x, train_data_y, valid_data_x, valid_data_y)
        print("验证集")
        score_valid = eval_model(valid_data_x, valid_data_y)
        macro_f1_score.append(score_valid)

    # 输出五折交叉验证结果
    print("五折交叉验证结果")
    print("f1_score均值： %.2f" % np.mean(macro_f1_score))
    print("f1_score方差： %.2f" % np.var(macro_f1_score))
    # 全训练集建模
    print("全训练集建模")
    model_data = pd.DataFrame(model_data.copy())
    model_label = pd.DataFrame(model_label.copy())
    valid_data = pd.DataFrame(test_data.copy())
    valid_label = pd.DataFrame(test_label.copy())
    gbm_model, evals_result = train_model(model_data, model_label, valid_data, valid_label)
    # 测试集评测
    print("测试集")    
    score_test = eval_model(test_data, test_label)