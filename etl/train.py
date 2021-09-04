import os, joblib, numpy as np
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Scores
from sklearn.metrics import recall_score

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
lead_sets_names = ('twelve_leads', 'six_leads', 'four_leads', 'three_leads', 'two_leads')


class ModelTrainer:
    """
    Base class to train the models.

    Input:
        - classifier: A sklearn-compatible classifier object. For instance: sklearn.ensemble.RandomForestClassifier().
        - classifier_name: A name you are giving to the model (string). This tag will be part of the .pkl name when
        saving.
        - lead_set: A tag specifying the number of leads (string). Options given in the tuple lead_sets_names.

    It creates a ModelTrainer() object which allows to set and view the elements of the training complex. That includes
    the hyperparameter grid (empty by default) as the key feature to modify.

    Default metric to fit and optimize hyperparameters is an imbalanced accuracy scorer to increase the impact of false
    negatives.
    """

    def __init__(self, classifier, classifier_name: str, lead_set: str = 'twelve_leads'):

        self.numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]
        )

        self.categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder())]
        )

        self.models_dict = {}

        self._classifier = classifier
        self._classifier_name = classifier_name

        self._grid = {}

        self.lead_set = lead_set
        lead_sets_dict = dict(zip(lead_sets_names, lead_sets))
        if self.lead_set not in lead_sets_names:
            raise KeyError('No set of leads called "' + self.lead_set + '"')
        else:
            self.leads = list(lead_sets_dict[self.lead_set])

        self.pipe = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', self.numeric_transformer, self.leads + ['Age']),
                    ('cat', self.categorical_transformer, ['Sex']),
                ])),
            ('classifier', self._classifier)])

    def get_hyperps(self):
        return self.pipe.get_params(deep=True)

    def show_leads(self):
        print(self.leads)

    def set_grid(self, grid=None):
        if type(grid) is dict:
            self._grid = grid
            print('Custom hyperparameter grid was set.')
        else:
            print('Attention: No hyperparameter grid was set. Training will use the default values.')

    def train(self, data_train, labels_train, n_jobs: int = None):
        self._labs = labels_train.columns
        self.models = list()
        for lab in self._labs:
            print('   Training for label ' + lab + '\n')
            self._model = RandomizedSearchCV(self.pipe,
                                             self._grid,
                                             scoring='recall',
                                             cv=None,  # None = 5-fold CV
                                             n_jobs=n_jobs, verbose=0)
            fitted = self._model.fit(data_train[self.leads + ['Age', 'Sex']], labels_train[lab])
            self.models.append(fitted)
        print('Training done.')

    def best_hyperps(self):
        for i in range(0, len(self.models)):
            print('Model for label ' + self._labs[i])
            print(self.models[i].best_params_)

    def save_models(self, folder='models', include_date=True):
        self._directory = os.getcwd() + '\\' + folder
        if not os.path.isdir(self._directory):
            os.mkdir(self._directory)
        for lab, model in zip(self._labs, self.models):
            self.models_dict[lab] = model
        dt = ''
        if include_date:
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._directory + '\\' + dt + '_' + self.lead_set + '_' + self._classifier_name + '.pkl'
        joblib.dump(self.models_dict, filename)
        print('Models saved on ' + self._directory)
