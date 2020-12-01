from read_data import connect4
from sklearn.model_selection import KFold
from data_label_num import data_label_num
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

        # 实例化一个逻辑回归对象
        log_reg = LogisticRegression(multi_class='ovr', solver='sag')

        # 传入数据训练
        log_reg.fit(train_data_x, train_data_y)

        valid_y_pred = log_reg.predict(valid_data_x)  # 预测分类号
        score_valid = accuracy_score(valid_data_y, valid_y_pred)
        print("验证集：",score_valid)

    test_y_pred = log_reg.predict(test_data)  # 预测分类号
    score_test = accuracy_score(test_label, test_y_pred)
    print("测试集：", score_test)



