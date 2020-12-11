import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from read_data import connect4
from data_label_num import data_label_num

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

        # 实例化一个逻辑回归对象
        log_reg = LogisticRegression(multi_class='ovr', solver='saga')

        # 传入数据训练
        log_reg.fit(train_data_x, train_data_y)

        valid_y_pred = log_reg.predict(valid_data_x)  # 预测分类号
        score_valid = f1_score(valid_data_y, valid_y_pred, average="macro")
        macro_f1_score.append(score_valid)
        print("验证集：",score_valid)

    # 输出五折交叉验证结果
    print("五折交叉验证结果")
    print("f1_score均值：", np.mean(macro_f1_score))
    print("f1_score方差：", np.var(macro_f1_score))
    # 全训练集建模
    model_data = pd.DataFrame(model_data.copy())
    model_label = pd.DataFrame(model_label.copy())
    log_reg.fit(model_data, model_label)
    # 测试集评测
    test_y_pred = log_reg.predict(test_data)  # 预测分类号
    score_test = f1_score(test_label, test_y_pred, average="macro")
    print("测试集：", score_test)