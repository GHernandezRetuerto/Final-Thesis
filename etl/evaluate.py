import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import scikitplot.metrics as skp


class EvalModel():

    def __init__(self, model_path: str, data_test, labels_test):
        self.models_dict = joblib.load(model_path)
        self.pred_dict = {}
        self._labels_test = labels_test
        for lab, model in zip(self.models_dict.keys(), self.models_dict.values()):
            prediction = model.predict(data_test)
            probability = model.predict_proba(data_test)
            pred_prob = [prediction, probability]
            self.pred_dict[lab] = pred_prob

    def get_labels(self):
        return self.pred_dict.keys()

    def plot_lift(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_cumulative_gain(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def plot_roc(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_roc_curve(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def plot_prec_rec(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_precision_recall(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def plot_calibration(self):
        probas_list = []
        for lab in self.models_dict.keys():
            probas_list.append(self.pred_dict[lab][1])
        skp.plot_precision_recall(self._labels_test[lab], probas_list, self.models_dict.keys())
        plt.title(lab)
        plt.show()

    def plot_ks(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_ks_statistic(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def plot_confusion(self):
        for lab in self.models_dict.keys():
            pred = self.pred_dict[lab][0]
            skp.plot_confusion_matrix(self._labels_test[lab], pred)
            plt.show()


# https://scikit-plot.readthedocs.io/en/stable/metrics.html
# https://www.justintodata.com/generate-reports-with-python/#html
