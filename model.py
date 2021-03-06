from tensorflow.keras import Input, models
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout, Conv1D, GRU, Reshape
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow_core.python.keras.layers import LeakyReLU, LSTM, GRU


def build_model(input, output_size, stateful=False, model_input=None):
    if model_input is None:
        model_input = input
    l = input

    # l = GRU(16, name="lstm_1", return_sequences=True, stateful=stateful)(l)

    # lstm = LSTM(16, name="lstm_2", return_state=True)
    #
    # states = []
    # outputs = []
    # state = None
    # for t in range(input.batch_shape[1]):
    #     input_t = input[:, t, :]
    #     input_t = tf.expand_dims(input_t, 1)
    #     output_t, h, c = lstm(input_t, initial_state=state)
    #     state = h, c
    #     states.append(state)
    #     outputs.append(output_t)
    #
    #
    # states = [tf.concat(x, axis=1) for x in states]
    # l = tf.stack(states, axis=1)
    # l = TimeDistributed(Dropout(0.4))(l)



    # l = Conv1D(32, 3, padding='SAME', activation='relu')(l)
    # l = TimeDistributed(Dropout(0.3))(l)

    l = GRU(128, return_sequences=True, stateful=stateful)(l)
    l = LeakyReLU()(l)
    l = Dropout(0.2)(l)

    l = GRU(128, return_sequences=True, stateful=stateful)(l)
    l = LeakyReLU()(l)
    l = Dropout(0.2)(l)

    l = TimeDistributed(Dense(64, activation=LeakyReLU()))(l)
    l = Dropout(0.2)(l)

    l = TimeDistributed(Dense(32, activation=LeakyReLU()))(l)
    l = Dropout(0.2)(l)

    l = TimeDistributed(Dense(output_size, activation='softmax', name='dense_1'))(l)

    return Model(inputs=model_input, outputs=l)

def test_model():
    model = models.Sequential()
    model.add(GRU(256, activation='tanh', input_shape=(29, 70)))
    model.add(Dense(5, activation='softmax')) # this was switched out of softmax
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=model.metrics + ["acc"], weighted_metrics=model.metrics + ["acc"])

    return model

def train_model(sequence_size, output_size=5):
    model = build_model(Input((sequence_size, 70)), output_size)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(), sample_weight_mode='temporal')

    return model


def inference_model(train_model: Model, output_size=5):
    inp = Input(batch_input_shape=(1, 70))
    l = Reshape((1, 70))(inp)
    model = build_model(l, output_size, stateful=True, model_input=inp)

    named_train_layers = {}
    for tl in train_model.layers:
        named_train_layers[tl.name] = tl

    for layer in model.layers:
        if layer.name in named_train_layers and len(layer.weights) > 0:
            layer.set_weights(named_train_layers[layer.name].get_weights())

    # for layer, train_layer in zip(model.layers[1:], train_model.layers):
    #     layer.set_weights(train_layer.get_weights())

    return model
