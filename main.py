from etl.train import ModelTrainer
from etl.load_data import LoadData
from etl.assets.grids import *


twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
lead_sets_names = ('twelve_leads', 'six_leads', 'four_leads', 'three_leads', 'two_leads')

data_path = r'C:\Users\ghdez\Mi unidad\MASTER UC3M\TFM\Repo\data\datasets\WFDB_CPSC2018_data.csv'
label_path = r'C:\Users\ghdez\Mi unidad\MASTER UC3M\TFM\Repo\data\datasets\WFDB_CPSC2018_labels.csv'

load = LoadData()
# load.load_data('data\\WFDB_ChapmanShaoxing', 'data\\datasets')
data, labels = load.get_csv(data_path, label_path)

data_train, data_test, labels_train, labels_test = load.split_train_test()

models_dict = {
    'LR': LogisticRegression(),
    'kNN': KNeighborsClassifier(),
    # 'SVM': SVC(),
    'NB': BernoulliNB(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(use_label_encoder=False, verbosity=0)
}
# Imported from etl/assets/grids.py
grids_dict = {
    'LR': grid_lr,
    'kNN': grid_knn,
    # 'SVM': grid_svm,
    'NB': grid_nb,
    'RF': grid_rf,
    'XGB': grid_xgb
}

for key in models_dict.keys():
    lead_set = 'twelve_leads'
    print('Training model: ' + key + ' for ' + lead_set)
    current_model = ModelTrainer(classifier=models_dict[key], classifier_name=key, lead_set=lead_set)
    current_model.set_grid(grids_dict[key])
    current_model.train(data_train, labels_train)
    current_model.save_models('models\\two_leads\\prueba' + '\n\n', include_date=False)
