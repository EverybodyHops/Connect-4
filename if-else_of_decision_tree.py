from sklearn.metrics import f1_score

from read_data import connect4
from decision_tree import decision_tree


def predict(feature):
    if feature[41] == "x":
        if feature[13] == "x":
            result = "draw"
        elif feature[13] == "o":
            result = "win"
        else:
            if feature[7] == "o":
                result = "loss"
            else:
                result = "draw"
    elif feature[41] == "o":
        if feature[18] == "x":
            result = "win"
        elif feature[18] == "o":
            if feature[38] == "o":
                result = "draw"
            else:
                result = "loss"
        else:
            if feature[12] == "o":
                result = "draw"
            else:
                result = "win"
    else:
        if feature[20] == "x":
            result = "win"
        elif feature[20] == "o":
            if feature[21] == "x":
                result = "win"
            else:
                result = "loss"
        else:
            if feature[19] == "o":
                result = "loss"
            else:
                result = "win"
    return result

if __name__ == "__main__":
    dataset = connect4()
    data = dataset.get_all_data()
    
    # 计算if-else在数据集上的正确率
    right = 0
    res_ifelse = []
    for i in range(len(data[0])):
        feature = data[0][i]
        pred = predict(feature)
        res_ifelse.append(pred)
        if pred == data[1][i]:
            right += 1
    print("if-else正确率：",right / len(data[0]))
    
    # 计算决策树在数据集上的正确率
    right = 0
    res_dt = []
    dt = decision_tree(max_depth=3)
    dt.train(dataset.get_train())
    for i in range(len(data[0])):
        pred = dt.predict(data[0][i])
        res_dt.append(pred)
        if pred == data[1][i]:
            right += 1
    print("决策树正确率：",right / len(data[0]))
    
    # 计算if-else和决策树结果之间的macro f1 score
    score = f1_score(res_dt, res_ifelse, average="macro")
    print("if-else与决策树之间的macro f1 score为：",score)