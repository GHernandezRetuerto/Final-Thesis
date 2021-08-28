import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics


import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics


class EvalModel():

    def __init__(self, model_path: str, data_test, labels_test):
        self.models_dict = joblib.load(model_path)
        self.pred_dict = {}
        for lab, model in zip(self.models_dict.keys(), self.models_dict.values()):
            prediction = model.predict(data_test)
            probability = model.predict_proba(data_test)
            pred_prob = [prediction, probability]
            self.pred_dict[lab] = pred_prob

    def get_labels(self):
        return self.pred_dict.keys()

    def plot_lift(self, y_pred, step=0.01):
        for lab in self.models_dict.keys()
            aux_lift = pd.DataFrame()
            aux_lift['real'] = labels_test[self.labels_test
            aux_lift['predicted'] = y_pred
            aux_lift.sort_values('predicted', ascending=False, inplace=True)
            x_val = np.arange(step, 1 + step, step)
            ratio_ones = aux_lift['real'].sum() / len(aux_lift)
            y_v = []










def plot_lift(y_val, y_pred, step=0.01):
    aux_lift = pd.DataFrame()
    aux_lift['real'] = y_val
    aux_lift['predicted'] = y_pred
    aux_lift.sort_values('predicted', ascending=False, inplace=True)
    x_val = np.arange(step, 1 + step, step)
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    y_v = []

    for x in x_val:
        num_data = int(
            np.ceil(x * len(aux_lift)))
        data_here = aux_lift.iloc[:num_data, :]
        ratio_ones_here = data_here['real'].sum() / len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)

        fig, axis = plt.subplots()
        fig.figsize = (10, 10)
        axis.plot(x_val, y_v, 'g-', linewidth=3, markersize=5)
        axis.plot(x_val, np.ones(len(x_val)), 'k-')
        axis.set_xlabel("Proportion of sample")
        axis.set_ylabel('Lift')
        plt.title('Lift Curve')
        plt.show()


def plot_ratio_ones_by_score(y_true, y_pred, salto=0.05):
    x_v = np.arange(salto, 1 + salto, salto)
    aux_lift = pd.DataFrame(columns=['real', 'pred'])

    aux_lift['real'] = y_true
    aux_lift['pred'] = y_pred
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    aux_lift = aux_lift.sort_values('pred', ascending=False)
    y_v = []
    for x in x_v:
        num_data_m_1 = int(np.ceil((x - salto) * len(aux_lift)))
        num_data = int(np.ceil(x * len(aux_lift)))
        data_here = aux_lift.iloc[num_data_m_1:num_data, :]
        ratio_ones_here = data_here['real'].sum() / len(data_here)
        y_v.append(ratio_ones_here)

    result = pd.DataFrame(columns=['pct_sample', 'ratio_ones'])
    result['pct_sample'] = x_v
    result['ratio_ones'] = y_v

    # plt.figure(figsize=(12,10))
    plt.plot(x_v, y_v, 'g.-')
    plt.plot(x_v, ratio_ones * np.ones(len(x_v)))
    plt.xlabel('Proportion of sample')
    plt.ylabel('Success ratio')
    plt.title('Success ratio by score')
    # plt.ylim((0,0.3))
    plt.show()
    print("******************Table ratios ******************")
    result


def plot_respt_acum_by_score(y_true, y_pred, salto=0.05):
    x_v = np.arange(salto, 1 + salto, salto)
    aux_lift = pd.DataFrame(columns=['real', 'pred'])

    aux_lift['real'] = y_true
    aux_lift['pred'] = y_pred
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    aux_lift = aux_lift.sort_values('pred', ascending=False)
    y_v = []
    for x in x_v:
        num_data = int(np.ceil(x * len(aux_lift)))
        data_here = aux_lift.iloc[0:num_data, :]
        ratio_ones_here = data_here['real'].sum() / len(data_here)
        y_v.append(ratio_ones_here)

    result = pd.DataFrame(columns=['pct_sample', 'respt_acum'])
    result['pct_sample'] = x_v
    result['respt_acum'] = y_v

    # plt.figure(figsize=(12,10))
    plt.plot(x_v, y_v, 'g.-')
    plt.plot(x_v, ratio_ones * np.ones(len(x_v)))
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    plt.xlabel('Proportion of sample')
    plt.ylabel('Accumulated response')
    plt.title('Accumulated response by score')
    # plt.ylim((0,0.3))
    plt.show()

    print("******************Table respt acum ******************")
    for row in range(result.shape[0]):
        print(result.iloc[row, :].values)


def plot_scores_probability(y_true, y_pred, salto=0.1):
    x_v = np.arange(min(y_pred), max(y_pred) + salto, salto)
    aux_df = pd.DataFrame(columns=['real', 'pred'])

    aux_df['real'] = y_true
    aux_df['pred'] = y_pred

    aux_df = aux_df.sort_values('pred', ascending=False)
    y_v = [0]
    score_v = [0]
    for i in range(len(x_v) - 1):
        aux_df_filter = aux_df[(aux_df.pred >= x_v[i]) & (aux_df.pred < x_v[i + 1])]
        ratio_unos = aux_df_filter.real.mean()
        ratio_score = aux_df_filter.pred.mean()
        y_v.append(ratio_unos)
        score_v.append(ratio_score)
    plt.plot(score_v, y_v, 'g.-')
    plt.plot(x_v, x_v, 'r--')
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    plt.xlabel('Score')
    plt.ylabel('Ones Ratio')
    plt.title('Scores vs Probabilities')
    # plt.ylim((0,0.3))
    plt.show()


def hist_scores(y_pred, salto=0.1):
    x_v = np.arange(min(y_pred), max(y_pred) + salto, salto)
    plt.hist(x=y_pred, bins=len(x_v) - 1, color='#0504aa',
             alpha=0.7, rwidth=0.85)
    plt.show()


def metricas(prediction, y_dev, target_ratio, probability):
    # METRICAS
    # target_ratio = np.mean(y_dev)
    print("-" * 80)
    print("RATIO 0/1 (TEST):", target_ratio)
    print('')

    # accuracy = accuracy_score(y_dev, prediction)
    # print("ACCURACY:", accuracy)
    # print("#-"*80)
    auc = roc_auc_score(y_dev, probability)
    print("ROC-AUC:", auc)
    print("-" * 80)
    gini = (2 * auc) - 1
    print("GINI:", gini)
    print("-" * 80)

    rmse = np.sqrt(1 / len(y_dev) * ((probability - y_dev) ** 2).sum())
    print("RMSE:", rmse)
    print("-" * 80)

    print("ROC CURVES")
    fpr, tpr, _ = metrics.roc_curve(y_dev, probability)
    auc = metrics.roc_auc_score(y_dev, probability)
    plt.plot(fpr, tpr, label="ROC CURVE, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()
    print("-" * 80)
    max_lift = 1 / target_ratio if 1 / target_ratio > 2 else 1 / (1 - target_ratio)
    print("THEORICAL MAX LIFT:", max_lift)
    print("-" * 80)
    print("LIFT PLOT")
    step_def = 0.1
    lift_step = plot_lift(y_dev, probability, step=step_def)
    print("Lift at {0}% is: {1}".format(step_def * 100, lift_step))
    print("-" * 80)

    print("FREQUENCY PLOT")
    plot_ratio_ones_by_score(y_dev, probability, salto=0.01)
    print("-" * 80)
    print("Accumulated response")
    plot_respt_acum_by_score(y_dev, probability, salto=0.01)
    print("-" * 80)
    print('Scores vs Probabilities')
    plot_scores_probability(y_dev, probability, salto=0.1)
    print("-" * 80)
    print('Scores Histogram')
    hist_scores(probability, salto=0.1)
