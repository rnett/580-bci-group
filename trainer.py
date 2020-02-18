import random

import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

import h5py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.callbacks import EarlyStopping

from commands import Command
from model import train_model

parser = ArgumentParser()

parser.add_argument("data_files", nargs="+", help="Data files to train on", default=[])
parser.add_argument("--output_file", "-o", type=str, help="Output file",
                    default=str(Path(f"./models/{datetime.now().strftime('%Y-%m-%d--%H_%M_%S')}--model")))
parser.add_argument("--epochs", "-e", type=int, help="Epochs to train for")
parser.add_argument("--steps_per_epoch", "-s", type=int, help="Steps/batches per epoch")
parser.add_argument("--sequence_length", "-sl", type=int, help="Size of the RNN sequence to train on", default=100)
parser.add_argument("--batch_size", "-b", type=int, help="Batch Size", default=32)
parser.add_argument("--validation_steps", "-vs", type=int, default=10, help="Validation steps per epoch (batches of "
                                                                            "batch_size sequence_size sequences to "
                                                                            "validate on)")
parser.add_argument("--test_split", "-t", type=float, default=0.1, help="Test split")

NOTHING_WEIGHT = 0.10


def random_segments(data, batch_size, segment_length: int, noise: bool):
    use_data = [d for d in data if d[0].shape[0] >= segment_length]
    while True:
        batch_features = []
        batch_labels = []
        weight = []
        for i in range(batch_size):
            features, labels = random.choice(use_data)
            start = random.randint(0, len(features) - (segment_length + 1))

            feature = features[start:start + segment_length]

            if noise:
                feature += np.random.normal(0, 0.2, feature.shape)

            batch_features.append(feature)
            l = labels[start:start + segment_length]
            batch_labels.append(l)
            # 0.1 if Nothing, else 1
            weight.append(l[:, 0] * -(1 - NOTHING_WEIGHT) + 1)

        yield np.stack(batch_features, axis=0), np.stack(batch_labels, axis=0), np.stack(weight, axis=0)


def all_segments(files, index: int, batch_size: int, segment_length: int):
    for f in files:
        data = f[index]
        i = 0
        while i < len(data) - segment_length:
            batch = []
            for j in range(batch_size):
                if i >= len(data) - segment_length:
                    break

                batch.append(data[i:i + segment_length])
                i += 20

            if len(batch) > 0:
                yield np.stack(batch, axis=0)


def plot_confusion_matrix(cm, target_names, title="Confusion Matrix"):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    fig.set_dpi(100)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=target_names, yticklabels=target_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.plot()
    plt.show()


def display_report(files, name, args):
    target_names = [c.name for c in list(Command)]
    y_true = list(all_segments(files, 1, args.batch_size, args.sequence_length))
    steps = len(y_true)
    y_true = np.concatenate(y_true, axis=0)
    y_true = np.argmax(y_true, axis=-1).flatten()

    Y_pred = model.predict(all_segments(files, 0, args.batch_size, args.sequence_length),
                           steps=steps)
    y_pred = np.argmax(Y_pred, axis=-1).flatten()
    # y_pred = y_pred.flatten()
    print(f'{name} Classification Report')
    print(classification_report(y_true, y_pred, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cm, target_names, f"{name} Confusion Matrix")


if __name__ == '__main__':

    if len(tf.config.list_physical_devices('GPU')) < 1:
        print("Warning: Not using GPU, this will be slow")

    args = parser.parse_args()

    data_files = [Path(d) for d in args.data_files]

    for d in data_files:
        if not d.exists():
            raise FileNotFoundError(f"Data file {d} does not exist")

    all_data = []

    for d in data_files:
        with h5py.File(str(d), 'r') as f:
            all_data.append((np.log(f["features"][:]), f["labels"][:]))

    all_features = np.concatenate([d[0] for d in all_data], axis=0)
    # all_features = np.log(all_features)

    f_mean = np.mean(all_features, axis=0)
    f_std = np.std(all_features, axis=0)

    shift = 2
    all_data = [((d[0] - f_mean) / f_std, np.concatenate([np.zeros((shift, 5)), d[1][:-shift, :]], axis=0)) for d in
                all_data]

    num_test = int(sum(d[0].shape[0] for d in all_data) * args.test_split)

    file_idx = 0
    data_idx = 0
    remaining = num_test

    train = []
    test = []

    for d in all_data:
        if remaining > 0:
            if d[0].shape[0] <= remaining:
                test.append(d)
                remaining -= d[0].shape[0]
            else:
                left = d[0].shape[0] - remaining
                if left >= args.sequence_length:
                    test.append((d[0][:remaining], d[1][:remaining]))
                    train.append((d[0][remaining:], d[1][remaining:]))
                    remaining = 0
                else:
                    test.append(d)
                    remaining = 0
        else:
            train.append(d)

    model = train_model(args.sequence_length, 5)
    model.compile(optimizer=model.optimizer,
                  loss=model.loss,
                  metrics=model.metrics + ["acc"], weighted_metrics=model.metrics + ["acc"],
                  sample_weight_mode=model.sample_weight_mode)

    hist = model.fit(random_segments(train, args.batch_size, args.sequence_length, False),
                     epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                     validation_data=random_segments(test, args.batch_size, args.sequence_length, False),
                     validation_steps=args.validation_steps,
                     # callbacks=[EarlyStopping(monitor='val_acc', mode='max', patience=10, restore_best_weights=True),]
                     )

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation accuracy values
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['acc_1'])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['val_acc_1'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', "Weighted Train", 'Test', "Weighted Test"], loc='upper left')
    plt.show()

    out_path = Path(args.output_file)

    model.save(args.output_file, include_optimizer=False)

    np.save(out_path / "mean.npy", f_mean)
    np.save(out_path / "std.npy", f_std)

    # just train
    display_report(train, "Just Train", args)

    # all labels
    display_report(all_data, "All Data", args)

    # just test
    display_report(test, "Just Test", args)
