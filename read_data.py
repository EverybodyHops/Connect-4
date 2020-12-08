import pandas as pd
import numpy as np

class connect4:
    def __init__(self, file_name="./connect-4.data", train_test_rate=0.9, numerical=False, one_hot=False):
        # numerical 变量是为将数据集读成数字形式来训练神经网络等模型预留的参数
        # one_hot 变量同理

        self.data = pd.read_csv(file_name, header=None).sample(frac=1, random_state=1111).reset_index(drop=True)
        self.data_num = self.data.shape[0]
        split_idx = int(self.data_num * train_test_rate)

        self.train_x = self.data.iloc[:split_idx, :-1].values.tolist()
        self.train_y = self.data.iloc[:split_idx, -1].values.tolist()
        self.test_x = self.data.iloc[split_idx:, :-1].values.tolist()
        self.test_y = self.data.iloc[split_idx:, -1].values.tolist()

    def get_all_data(self):
        return self.data

    def get_train(self):
        return self.train_x, self.train_y

    def get_test(self):
        return self.test_x, self.test_y

if __name__ == "__main__":
    dataset = connect4()
    print(dataset.get_train()[1][0:40])