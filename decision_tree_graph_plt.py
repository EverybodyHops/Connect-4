from read_data import connect4
from decision_tree import decision_tree
# from pyecharts import options as opts
import  pyecharts.options   as   opts
from pyecharts.charts import Page, Tree



def dt_plt(dt, node, max_depth, layer):
    son = dt.node_list[node].son
    data_temp = dict()
    data_temp["name"] = dt.node_list[node].feature_idx
    if dt.node_list[node].category == "win":
        data_temp["value"] = 1
    elif dt.node_list[node].category == "loss":
        data_temp["value"] = 0
    elif dt.node_list[node].category == "draw":
        data_temp["value"] = 2
    if layer == max_depth:
        pass
    if son:
        son_len = len(son)
        if max_depth> layer:
            data_temp["children"] = []
            for each in son:
                data_temp["children"].append({"name":each,"children":[dt_plt(dt, son[each], max_depth, layer+1)]})

            if son_len == 2:
                if "b" not in son:
                    data_temp["children"].append(
                        {"name": "b", "children": [{"name": dt.node_list[node].category}]})
                if "x" not in son:
                    data_temp["children"].append(
                        {"name": "x", "children": [{"name": dt.node_list[node].category}]})
                if "o" not in son:
                    data_temp["children"].append(
                        {"name": "o", "children": [{"name": dt.node_list[node].category}]})

        else:
            data_temp["children"] = []
            for each in son:
                data_temp["children"].append({"name": each, "children": [{"name": dt.node_list[son[each]].category}]})
            if son_len == 2:
                if "b" not in son:
                    data_temp["children"].append(
                        {"name": "b", "children": [{"name": dt.node_list[node].category}]})
                if "x" not in son:
                    data_temp["children"].append(
                        {"name": "x", "children": [{"name": dt.node_list[node].category}]})
                if "o" not in son:
                    data_temp["children"].append(
                        {"name": "o", "children": [{"name": dt.node_list[node].category}]})

    else:
        data_temp["name"] = dt.node_list[node].category
        return data_temp
    return data_temp
    # data_temp = dict()
    # data_temp["name"] = dt.node_list[node].feature_idx
    # for each in son:
    #     data_son_temp = dict()
    #     data_son_temp["name"] = each
    #     data_son_temp["children"] = []
    #     data_son_temp["children"].append(dt_plt(dt, son[each]))
    # pass
if __name__ == "__main__":
    dataset = connect4(numerical=False, one_hot=False)
    max_depth = 7
    dt = decision_tree(max_depth=max_depth)
    dt.train(dataset.get_train())
    data_p = [dt_plt(dt, 0, max_depth, 1)]
    # print("//////")
    tree = (
        Tree(init_opts=opts.InitOpts(width="10000px", height="1000px"))
            .add("", data_p, orient="TB")
            .set_global_opts(title_opts=opts.TitleOpts(title="大作业三决策树"))
    )

    tree.render()
