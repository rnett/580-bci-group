import h5py
import numpy as np
from tensorflow_core.python.keras.models import load_model

from trainer import segment_data, display_report

model = load_model("best_model")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=model.metrics + ["acc"], weighted_metrics=model.metrics + ["acc"])

data = h5py.File("class.hdf5")

data = [(np.log10(data["features"]), data["labels"]),]

new_data, new_labels = segment_data(data)

display_report(model, new_data, new_labels, "New Test", {})