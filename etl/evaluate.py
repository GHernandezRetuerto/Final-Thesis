import joblib, re
import matplotlib.pyplot as plt
import numpy as np
import scikitplot.metrics as skp
import sklearn.metrics as skl


class EvalModel:

    def __init__(self, model_path: str, data_test, labels_test, proba_threshold=0.5):
        self.models_dict = joblib.load(model_path)
        self.pred_dict = {}
        self._labels_test = labels_test
        for lab, model in zip(self.models_dict.keys(), self.models_dict.values()):
            probability = model.predict_proba(data_test)
            prediction = np.where(probability[:, 1] < proba_threshold, False, True)
            pred_prob = [prediction, probability]
            self.pred_dict[lab] = pred_prob

    def best_hyperps(self):
        for i in range(0, len(self.models_dict)):
            print('Model for label ' + self._labs[i])
            print(self.models_dict[i].best_params_)

    def get_labels(self):
        return self.pred_dict.keys()

    def plot_lift(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_lift_curve(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def plot_cumul_gains(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_cumulative_gain(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def plot_roc(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_roc(self._labels_test[lab], prob, plot_micro=False)
            plt.title(lab)
            plt.show()

    def plot_calibration(self):
        for lab in self.models_dict.keys():
            prob = list(self.pred_dict[lab][1])
            skp.plot_calibration_curve(self._labels_test[lab], prob)
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
            plt.figure(figsize=(10, 10))
            skp.plot_confusion_matrix(self._labels_test[lab], pred)
            plt.title(lab)
            plt.show()

    def plot_prec_rec(self):
        for lab in self.models_dict.keys():
            prob = self.pred_dict[lab][1]
            skp.plot_precision_recall(self._labels_test[lab], prob)
            plt.title(lab)
            plt.show()

    def balanced_accuracy(self):
        for lab in self.models_dict.keys():
            pred = self.pred_dict[lab][0]
            acc = skl.balanced_accuracy_score(self._labels_test[lab], pred)
            print(lab + '\n' + str(acc) + '\n')

    def f_beta(self, beta=0):
        if beta < 0:
            raise KeyError('Invalid value of beta, select a value between 0 (pure precision) and inf (pure recall).')
        print('beta = ' + str(beta))
        for lab in self.models_dict.keys():
            pred = self.pred_dict[lab][0]
            acc = skl.fbeta_score(self._labels_test[lab], pred, beta=beta)
            print(lab + '\n' + str(acc) + '\n')

    def accuracy(self):
        for lab in self.models_dict.keys():
            pred = self.pred_dict[lab][0]
            acc = skl.accuracy_score(self._labels_test[lab], pred)
            print(lab + '\n' + str(acc) + '\n')

    def recall(self):
        for lab in self.models_dict.keys():
            pred = self.pred_dict[lab][0]
            acc = skl.recall_score(self._labels_test[lab], pred)
            print(lab + '\n' + str(acc) + '\n')

    def create_report(self):
        input_dict = dict(zip(range(1, len(self.models_dict.keys())+1), self.models_dict.keys()))
        print(input_dict)
        label = int(input('Choose between the options above (input number): \n'))
        print(list(self.models_dict.keys())[label-1])

    def get_model_params(self):
        for lab, model in zip(self.models_dict.keys(), self.models_dict.values()):
            print('############ ' + lab + '############')
            params = model.best_params_
            for param in list[filter(re.match, params.keys())]:
                print('\n' + param + ': ' + params[param] + '\n')
                print()

# https://scikit-plot.readthedocs.io/en/stable/metrics.html
# https://www.justintodata.com/generate-reports-with-python/#html

