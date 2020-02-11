from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout, Conv1D, GRU, Reshape
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.regularizers import l2


def build_model(input, output_size, stateful=False, model_input=None):
    if model_input is None:
        model_input = input

    l = LSTM(16, name="lstm_1", return_sequences=True, stateful=stateful)(input)
    # l = LSTM(6, name="lstm_2", return_sequences=True, stateful=stateful)(l)

    # l = TimeDistributed(Dropout(0.6))(l)
    # l = TimeDistributed(Dense(16, activation='relu'))(l)
    l = TimeDistributed(Dropout(0.2))(l)

    l = TimeDistributed(Dense(output_size, activation='softmax', name='dense_1', kernel_regularizer=l2(0.01)))(l)

    return Model(inputs=model_input, outputs=l)


def train_model(sequence_size, output_size=5):
    model = build_model(Input((sequence_size, 70)), output_size)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.002, beta_1=0.87), sample_weight_mode='temporal')

    return model


def inference_model(train_model: Model, output_size=5):
    inp = Input(batch_input_shape=(1, 70))
    l = Reshape((1, 70))(inp)
    model = build_model(l, output_size, True, model_input=inp)

    for layer, train_layer in zip(model.layers[1:], train_model.layers):
        layer.set_weights(train_layer.get_weights())

    return model
