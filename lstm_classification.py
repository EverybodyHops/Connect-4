import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import variable

from lstm_model import RNN
from read_data import connect4
from data_label_num import data_label_num

EPOCH = 1
BATCH_SIZE = 32
TIME_STEP = 6
INPUT_SIZE = 7
LR = 0.01

if torch.cuda.is_available():
    USE_GPU = True
else:
    USE_GPU = False

if __name__ == '__main__':
    dataset = connect4(numerical=True, one_hot=False)
    _, _, train_dataset_num = data_label_num(dataset.train_x, dataset.train_y)
    test_data, test_label, _ = data_label_num(dataset.test_x, dataset.test_y)
    if USE_GPU:
        test_data_x = variable(test_data).cuda()
        test_data_y = variable(test_label).cuda()
    else:
        test_data_x = variable(test_data)
        test_data_y = variable(test_label)

    rnn = RNN()
    if USE_GPU:
        rnn.cuda()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    macro_f1_score = []
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    for train_index, valid_index in kf.split(train_dataset_num):
        train_data = pd.DataFrame(train_dataset_num.copy()).drop(valid_index)
        valid_data = pd.DataFrame(train_dataset_num.copy()).drop(train_index)

        train_data_y = train_data.iloc[:,-1].tolist()
        valid_data_y = valid_data.iloc[:,-1].tolist()
        train_data_x = train_data.drop(axis=1, columns=42).values.tolist()
        valid_data_x = valid_data.drop(axis=1, columns=42).values.tolist()

        train_data = Data.TensorDataset(torch.tensor(train_data_x), torch.tensor(train_data_y))
        train_data_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        if USE_GPU:
            valid_data_x = variable(valid_data_x).cuda()
            valid_data_y = variable(valid_data_y).cuda()
        else:
            valid_data_x = variable(valid_data_x)
            valid_data_y = variable(valid_data_y)

        for epoch in range(EPOCH):
            for step, (x, y) in enumerate(train_data_loader):
                if USE_GPU:
                    train_data_x = variable(x.view(-1, TIME_STEP, INPUT_SIZE)).cuda()
                    train_data_y = variable(y).cuda()
                else:
                    train_data_x = variable(x.view(-1, TIME_STEP, INPUT_SIZE))
                    train_data_y = variable(y)
                output = rnn(train_data_x.float())
                loss = loss_func(output, train_data_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 50 == 0:
                    print("Epoch :", epoch, "| train loss: %.4f" % loss.item())

        valid_output = rnn(valid_data_x.view(-1, TIME_STEP, INPUT_SIZE).float())
        if USE_GPU:
            pred_y = torch.max(valid_output, 1)[1].cuda().data.squeeze()
        else:
            pred_y = torch.max(valid_output, 1)[1].data.squeeze()
        
        score_valid = f1_score(valid_data_y, pred_y, average="macro")
        macro_f1_score.append(score_valid)
        print("验证集： %.2f" % score_valid)

    # 模型保存
    torch.save(rnn.state_dict(), '.\model')

    # 输出五折交叉验证结果
    print("五折交叉验证结果")
    print("f1_score均值： %.2f" % np.mean(macro_f1_score))
    print("f1_score方差： %.2f" % np.var(macro_f1_score))
    
    # 模型加载
    model = RNN()
    if USE_GPU:
        model.cuda()
    model.load_state_dict(torch.load('.\model'))
    test_output = model(test_data_x.view(-1, TIME_STEP, INPUT_SIZE).float())
    if USE_GPU:
        pred_y_test = torch.max(test_output, 1)[1].cuda().data.squeeze()
    else:
        pred_y_test = torch.max(test_output, 1)[1].data.squeeze()
    score_test = f1_score(test_data_y, pred_y_test, average="macro")
    print("测试集： %.2f" % score_test)

