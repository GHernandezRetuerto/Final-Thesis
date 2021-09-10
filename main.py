from etl.train import ModelTrainer
from etl.load_data import LoadData
from etl.assets.grids import grids_dict, models_dict

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
data_train, data_test, labels_train, labels_test = load.split_train_test()  # Random state = 23

# Classifier-wise
for key in models_dict.keys():
    lead_set = 'twelve_leads'
    print('Training model: {0} for {1}'.format(key, lead_set))
    current_model = ModelTrainer(classifier=models_dict[key],
                                 classifier_name=key,
                                 lead_set=lead_set,
                                 pca_n_components=None)
    current_model.set_grid(grids_dict[key])
    current_model.train(data_train, labels_train)
    current_model.save_models('models\\two_leads\\prueba', include_date=False)

# Lead-wise
for lead_set in lead_sets_names:
    key = 'RF'
    print('Training model: {0} for {1}'.format(key, lead_set))
    current_model = ModelTrainer(classifier=models_dict[key],
                                 classifier_name=key,
                                 lead_set=lead_set)
    current_model.set_grid(grids_dict[key])
    current_model.train(data_train, labels_train)
    current_model.save_models('models\\RF', include_date=False)

# PCA
key = 'RF'
lead_set = 'twelve_leads'
print('Training model: {0} for {1} using PCA'.format(key, lead_set))
current_model = ModelTrainer(classifier=models_dict[key],
                             classifier_name=key,
                             lead_set=lead_set,
                             pca_n_components=6)
current_model.set_grid(grids_dict[key])
current_model.train(data_train, labels_train)
current_model.save_models('models\\PCA', include_date=False)
