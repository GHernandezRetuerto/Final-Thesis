import os, sys
import pandas as pd
from helper_functions import *

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def load_data(data_directory, dataset_directory):
    # Save the name of the folder where the data comes from - later used when saving the data as .csv
    dataset_name = os.getcwd() + data_directory.split('\\')[-1]

    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the dataset if it does not already exist.
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)

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
    data_path = dataset_directory + '\\' + dataset_name + '_data.csv'
    data_df.to_csv(data_path, index=False)

    labels_df = pd.DataFrame(data=labels, columns=list(classes))
    labels_path = dataset_directory + '\\' + dataset_name + '_labels.csv'
    labels_df.to_csv(labels_path, index=False)


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the data and dataset folders as arguments, e.g., python load_data.py data dataset.')

    data_directory = sys.argv[1]
    dataset_directory = sys.argv[2]
    load_data(data_directory, dataset_directory)

    print('Done.')
