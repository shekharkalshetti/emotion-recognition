import os
from data.preprocess import create_data_path_csv
from data.prepare_data import prepare_data
from model.train import train_model


def main():
    ravdess_path = 'path/to/ravdess/'
    crema_path = 'path/to/crema/'
    tess_path = 'path/to/tess/'
    savee_path = 'path/to/savee/'

    create_data_path_csv(ravdess_path, crema_path, tess_path, savee_path)

    # Ensure you have this csv created with necessary features
    features_csv = 'path/to/your/feature_csv.csv'
    x_train, x_test, y_train, y_test, encoder = prepare_data(features_csv)

    model, accuracy = train_model(x_train, y_train, x_test, y_test)
    print(f'Model training complete with accuracy: {accuracy}')


if __name__ == '__main__':
    main()
