from etl.train import ModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
lead_sets_names = ('twelve_leads', 'six_leads', 'four_leads', 'three_leads', 'two_leads')


grids_dict = {
    'LR': {},
    'kNN': KNeighborsClassifier(),
    'NB': BernoulliNB(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(use_label_encoder=False)
}

models_dict = {
    'LR': LogisticRegression(),
    'kNN': KNeighborsClassifier(),
    'NB': BernoulliNB(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(use_label_encoder=False)
}

for key in models_dict.keys():
    current_model = ModelTrainer(classifier=models_dict[key], classifier_name=key, lead_set='two_leads')
    current_model.set_grid()
    current_model.train(data_train, labels_train)
    current_model.save_models('models\\pruebas')

