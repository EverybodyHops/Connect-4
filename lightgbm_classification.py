from read_data import connect4
from sklearn.model_selection import KFold
from data_label_num import data_label_num
import pandas as pd
from lightgbm_model import train_model, eval_model

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

        gbm_model, evals_result = train_model(train_data_x, train_data_y, valid_data_x, valid_data_y)
        print("验证集")
        score_valid = eval_model(valid_data_x, valid_data_y)

        #  测试集评测
    print("测试集")
    score_test = eval_model(test_data, test_label)



