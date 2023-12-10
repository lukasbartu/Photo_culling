__author__ = 'Lukáš Bartůněk'

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Input, Dropout
from keras import Sequential
from keras.optimizers.experimental import RMSprop
from keras.metrics import F1Score
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping

train_data = pd.read_csv("train_data.csv", header=None, on_bad_lines='skip')
validate_data = pd.read_csv("validate_data.csv", header=None)
test_data = pd.read_csv("test_data.csv", header=None)
train_results = pd.read_csv("train_results.csv", header=None, on_bad_lines='skip')
validate_results = pd.read_csv("validate_results.csv", header=None)
test_results = pd.read_csv("test_results.csv", header=None)

train_data = train_data.astype('float32')
train_data = train_data.to_numpy()
train_results = train_results.astype('float32')
train_results = train_results.to_numpy()

validate_data = validate_data.astype('float32')
validate_data = validate_data.to_numpy()
validate_results = validate_results.astype('float32')
validate_results = validate_results.to_numpy()

test_data = test_data.astype('float32')
test_data = test_data.to_numpy()
test_results = test_results.astype('float32')
test_results = test_results.to_numpy()

true_samples = 0
false_samples = 0
for result in train_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights = [(1 / true_samples) * (len(train_results) / 2.0), (1 / false_samples) * (len(train_results) / 2.0)]

true_samples = 0
false_samples = 0
for result in validate_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights_validate = np.asarray([(1 / true_samples) * (len(validate_results) / 2.0), (1 / false_samples) * (len(validate_results) / 2.0)])
val_sample_weights = validate_results*class_weights_validate[0] + (1-validate_results)*class_weights_validate[1]

true_samples = 0
false_samples = 0
for result in test_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights_test = [(1 / true_samples) * (len(test_results) / 2.0), (1 / false_samples) * (len(test_results) / 2.0)]

model = Sequential()
model.add(Input((162,)))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=0.00001), loss="binary_crossentropy",
              metrics=F1Score(threshold=0.5), weighted_metrics=[])

history = model.fit(train_data, train_results, epochs=400, batch_size=10, workers=-1,
                    validation_data=(validate_data, validate_results, val_sample_weights),
                    class_weight={0: class_weights[1], 1: class_weights[0]},
                    callbacks=[EarlyStopping(patience=100, restore_best_weights=True)])

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")

plt.title('Neural network optimization')
plt.ylabel('Cost/Total loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig("NN_training.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.plot(np.arange(100, len(history.history["val_loss"])-1, 1), history.history["loss"][100:-1], label="Training loss")
plt.plot(np.arange(100, len(history.history["val_loss"])-1, 1), history.history["val_loss"][100:-1], label="Validation loss")

plt.title('Close up of NN optimization')
plt.ylabel('Cost/Total loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig("NN_training_close-up.pdf", format="pdf", bbox_inches="tight")
plt.show()

res = model.evaluate(test_data, test_results)
with open('F1_score.txt', 'w') as file:
    file.write(str(res[1]))

cross_validation = np.concatenate((train_data, validate_data, test_data))
cross_validation_results = np.concatenate((train_results, validate_results, test_results))

res = model.evaluate(cross_validation, cross_validation_results)

model.save("best_nn_model.keras")
