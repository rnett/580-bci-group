from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Model


def train_model():
    input = Input((100, 70))
    l = LSTM(16, name="lstm_1", return_sequences=True)(input)
    l = LSTM(1, name="lstm_2", return_sequences=True)(l)

    model = Model(inputs=input, outputs=l)

    return model


def inference_model(train_model: Model):
    input = Input((1, 70))
    l = LSTM(16, name="lstm_1", return_sequences=True, stateful=True)(input)
    l = LSTM(1, name="lstm_2", return_sequences=True, stateful=True)(l)

    model = Model(inputs=input, outputs=l)

    model.get_layer("lstm_1").set_weights(train_model.get_layer("lstm_1").get_weights())
    model.get_layer("lstm_2").set_weights(train_model.get_layer("lstm_2").get_weights())

    return model
