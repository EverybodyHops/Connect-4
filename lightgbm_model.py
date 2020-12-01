import lightgbm as lgb
from sklearn.metrics import accuracy_score

# 参数设置
parameters = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_leaves': 120,
    'min_data_in_leaf': 100,
    'learning_rate': 0.06,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.4,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.2,
    'verbose': -1,
}


def train_model(data_train, label_train, data_valid, label_valid):
    lgb_train = lgb.Dataset(data_train, label_train)
    lgb_valid = lgb.Dataset(data_valid, label_valid, reference=lgb_train)

    print('Starting training...')
    # 模型训练
    evals_result = {}  # 记录训练结果所用
    gbm_model = lgb.train(parameters,
                          lgb_train,
                          valid_sets=[lgb_train, lgb_valid],
                          num_boost_round=200,  # 提升迭代的次数
                          early_stopping_rounds=50,
                          evals_result=evals_result,
                          verbose_eval=50
                          )

    print('Saving model...')
    # 模型保存
    gbm_model.save_model('model.txt')
    print('Done!')

    return gbm_model, evals_result


def eval_model(data_test, label_test):
    # 模型加载
    gbm_model = lgb.Booster(model_file='model.txt')

    print('Starting predicting...')
    # 模型预测
    label_test_pred = gbm_model.predict(data_test, num_iteration=gbm_model.best_iteration)
    label_test_pred_index = [list(x).index(max(x)) for x in label_test_pred]

    # 模型评估
    score = accuracy_score(label_test, label_test_pred_index)

    print(score)
    print('Done!')

    return score