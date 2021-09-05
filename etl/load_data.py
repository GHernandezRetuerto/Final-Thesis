import os, sys
import pandas as pd
from etl.helper_functions import *
from sklearn.model_selection import train_test_split

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


class LoadData:

    def __init__(self):
        return

    def load_data(self, raw_data_path, dataset_path):
        self.dataset_path = dataset_path
        self.raw_data_path = raw_data_path

        dataset_directory = self.dataset_path
        raw_data_directory = self.raw_data_path

        # Save the name of the folder where the data comes from - later used when saving the data as .csv
        dataset_name = os.getcwd() + raw_data_directory.split('\\')[-1]

        # Find header and recording files.
        print('Finding header and recording files...')

        header_files, recording_files = find_challenge_files(raw_data_directory)
        num_recordings = len(recording_files)

        if not num_recordings:
            raise NameError('No data was provided.')

        # Create a folder for the dataset if it does not already exist.
        if not os.path.isdir(dataset_path):
            os.mkdir()

        # Extract the classes from the dataset.
        print('Extracting classes...')

        classes = set()
        for header_file in header_files:
            header = load_header(header_file)
            classes |= set(get_labels(header))
        if all(is_integer(x) for x in classes):
            classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
        else:
            classes = sorted(classes)  # Sort classes alphanumerically if not numbers.
        num_classes = len(classes)

        # Extract the features and labels from the dataset.
        print('Extracting features and labels...')

        data = np.zeros((num_recordings, 14),
                        dtype=np.float32)
        labels = np.zeros((num_recordings, num_classes), dtype=np.bool)  # One-hot encoding of classes

        for i in range(num_recordings):
            print('    {}/{}...'.format(i + 1, num_recordings))

            # Load header and recording.
            header = load_header(header_files[i])
            recording = load_recording(recording_files[i])

            # Get age, sex and root mean square of the leads.
            age, sex, rms = get_features(header, recording, twelve_leads)
            data[i, 0:12] = rms
            data[i, 12] = age
            data[i, 13] = sex

            current_labels = get_labels(header)
            for label in current_labels:
                if label in classes:
                    j = classes.index(label)
                    labels[i, j] = 1

        # Save the current datasets of features and labels
        data_df = pd.DataFrame(data=data, columns=list(twelve_leads) + ['Age', 'Sex'])
        self.data_path = dataset_directory + '\\' + dataset_name + '_data.csv'
        data_df.to_csv(self.data_path, index=False)
        print('Features dataset saved on: ' + self.data_path + '\n')

        labels_df = pd.DataFrame(data=labels, columns=list(classes))
        self.labels_path = dataset_directory + '\\' + dataset_name + '_labels.csv'
        labels_df.to_csv(self.labels_path, index=False)
        print('Labels dataset saved on: ' + self.labels_path + '\n')

        print('Extraction done.')

    def get_csv(self, alt_data_path: str = None, alt_labels_path: str = None):
        if alt_data_path is not None:
            data_path = alt_data_path
        else:
            data_path = self.data_path
        if alt_labels_path is not None:
            labels_path = alt_labels_path
        else:
            labels_path = self.labels_path

        print("Fetching the CSV from\n'" +
              labels_path + "'\n and \n'" +
              data_path + "' dataset.\n")
        self.data = pd.read_csv(data_path)
        self.labels = pd.read_csv(labels_path)

        return self.data, self.labels

    def split_train_test(self, test_size: float = 0.25, random_state=23):
        print('Separating between train and test samples, test size = ' + str(100 * test_size) + '%.')
        data = self.data
        labels = self.labels
        return train_test_split(data, labels, test_size=test_size, random_state=random_state)
