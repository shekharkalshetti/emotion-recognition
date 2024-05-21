from .model import create_model
from sklearn.metrics import accuracy_score


def train_model(x_train, y_train, x_test, y_test, batch_size=64, epochs=100):
    model = create_model(input_shape=(x_train.shape[1], 1))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_test, y_test))

    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy
