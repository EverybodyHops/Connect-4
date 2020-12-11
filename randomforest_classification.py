import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

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

        n_features = 42  # 每个样本有几个属性或特征
        randomforest_model = RandomForestClassifier(n_estimators=10, max_features=int(math.sqrt(n_features)), max_depth=None,
                                      min_samples_split=2, bootstrap=True)
        randomforest_model.fit(train_data_x, train_data_y)

        valid_y_pred = randomforest_model.predict(valid_data_x)  # 预测分类号
        score_valid = f1_score(valid_data_y, valid_y_pred, average="macro")
        macro_f1_score.append(score_valid)
        print("验证集：", score_valid)

    # 输出五折交叉验证结果
    print("五折交叉验证结果")
    print("f1_score均值：", np.mean(macro_f1_score))
    print("f1_score方差：", np.var(macro_f1_score))
    # 全训练集建模
    model_data = pd.DataFrame(model_data.copy())
    model_label = pd.DataFrame(model_label.copy())
    randomforest_model.fit(model_data, model_label)
    # 测试集评测
    test_y_pred = randomforest_model.predict(test_data)  # 预测分类号
    score_test = f1_score(test_label, test_y_pred, average="macro")
    print("测试集：", score_test)