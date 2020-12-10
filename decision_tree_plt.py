from read_data import connect4
import numpy as np

class node:
    def __init__(self, idx):
        self.category = None
        self.x = None
        self.y = None
        self.son = {}
        self.father = None
        self.used = []
        self.e = 0.0
        self.idx = idx
        self.feature_idx = None
        self.son_feature_idx = {}

    def set_data(self, x, y):
        self.x = x
        self.y = y
        y_value_list = set([i for i in y])
        s = len(y)
        if s > 0:
            for v in y_value_list:
                t = y.count(v)
                if t > 0:
                    p = t / s
                    self.e -= p * np.log2(p)
        print("node", self.idx, "init, entropy is", self.e, "father is", self.father)

class decision_tree:
    def __init__(self, max_depth=5):
        self.root = None
        self.node_list = [node(i) for i in range((3 ** max_depth) * 2)]
        self.feature_num = 0
        self.max_depth = max_depth
        self.now = 0

    def train(self, data):
        def if_pure(root):
            if len(self.node_list[root].y) == 0:
                return True
            else:
                t = self.node_list[root].y[0]
                for l in self.node_list[root].y:
                    if l != t:
                        return False
                return True

        def most(root):
            if len(self.node_list[root].y) == 0:
                return self.node_list[root.father].category
            m = -1
            ret = None
            value_list = set([i for i in self.node_list[root].y])
            for v in value_list:
                t = self.node_list[root].y.count(v)
                if t > m:
                    m = t
                    ret = v
            return ret

        def IGR(root, feature_idx):
            x_value_list = set([i[feature_idx] for i in self.node_list[root].x])
            y_value_list = set([i for i in self.node_list[root].y])

            total = len(self.node_list[root].x)
            t = {}
            for x in x_value_list:
                t[x] = {}
                for y in y_value_list:
                    t[x][y] = 0

            for i in range(total):
                t[self.node_list[root].x[i][feature_idx]][self.node_list[root].y[i]] += 1
            # print(t)
            con_e = 0.
            h = 0.0
            for x in x_value_list:
                s = 0
                for y in y_value_list:
                    s += t[x][y]
                if s > 0:
                    tp = s / total
                    h -= tp * np.log2(tp)
                    for y in y_value_list:
                        if t[x][y] > 0:
                            p = t[x][y] / s
                            con_e -= tp * p * np.log2(p)
            if h == 0:
                h = 1.0
            return ((self.node_list[root].e - con_e) / h)

        def split(node_plt, root, now_depth, max_depth):
            self.node_list[root].category = most(root)
            if now_depth == max_depth or if_pure(root):
                return

            m = 0.0
            use_feature = -1
            for i in range(self.feature_num):
                if i not in self.node_list[root].used:
                    t = IGR(root, i)
                    if t > m:
                        m = t
                        use_feature = i
            print("split with feature", use_feature)
            if use_feature >= 0:
                self.node_list[root].feature_idx = use_feature
                x_value_list = set([i[use_feature] for i in self.node_list[root].x])
                t = {}
                for x in x_value_list:
                    t[x] = [[], []]
                total = len(self.node_list[root].x)
                for i in range(total):
                    t[self.node_list[root].x[i][use_feature]][0].append(self.node_list[root].x[i])
                    t[self.node_list[root].x[i][use_feature]][1].append(self.node_list[root].y[i])
                for x in x_value_list:
                    self.node_list[root].son[x] = self.now
                    self.node_list[self.now].used.append(use_feature)
                    self.node_list[self.now].father = root
                    self.node_list[self.now].set_data(t[x][0], t[x][1])
                    self.now += 1
                for x in x_value_list:
                    split(node_plt, self.node_list[root].son[x], now_depth + 1, max_depth)

                try:
                    # self.node_list[self.node_list[root].father].son_feature_idx[root] = use_feature
                    node_plt_temp = dict()
                    node_plt_temp["name"] = use_feature
                    node_plt_temp["children"] = []
                    node_plt["children"].append(node_plt_temp)
                except Exception as e:
                    print(e)

        self.feature_num = len(data[0][0])
        self.root = self.now
        self.now += 1
        self.node_list[self.root].set_data(data[0], data[1])
        most(self.root)
        node_plt = dict()
        split(node_plt, self.root, 0, self.max_depth)

    def predict(self, x):
        k = self.root
        while True:
            if self.node_list[k].feature_idx is not None:
                t = x[self.node_list[k].feature_idx]
                if t in self.node_list[k].son:
                    k = self.node_list[k].son[t]
                else:
                    break
            else:
                break
        return self.node_list[k].category

if __name__ == "__main__":
    dataset = connect4(numerical=False, one_hot=False)
    dt = decision_tree(max_depth=3)
    dt.train(dataset.get_train())
    print("//////")
    # test = dataset.get_test()
    # right = 0
    # for i in range(len(test[1])):
    #     res = dt.predict(test[0][i])
    #     if res == test[1][i]:
    #         right += 1
    # print(right / len(test[1]))