import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def prepare_data(features_csv):
    Features = pd.read_csv(features_csv)
    X = Features.iloc[:, :-1].values
    Y = Features['labels'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, random_state=0, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    return x_train, x_test, y_train, y_test, encoder
