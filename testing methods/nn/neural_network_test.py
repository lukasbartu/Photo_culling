__author__ = 'Lukáš Bartůněk'

import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Input, Dropout
from keras import Sequential
from keras.optimizers.experimental import RMSprop
from keras.metrics import F1Score
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

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

model = Sequential()
model.add(Input((162,)))
model.add(Dropout(0.5))
model.add(Dense(80))
model.add(Dropout(0.5))
model.add(Dense(40))
model.add(Dense(1, activation='sigmoid'))

# best_checkpoint_path = "best_checkpoint_nn"
#
# save_best_model = ModelCheckpoint(best_checkpoint_path, monitor='val_f1_score',
#                                   save_best_only=True, save_weights_only=True, mode="max")

model.compile(optimizer=RMSprop(learning_rate=0.0001), loss="binary_crossentropy", metrics=F1Score(threshold=0.5))

true_samples = 0
false_samples = 0
for result in train_results:
    if result == 1:
        true_samples += 1
    else:
        false_samples += 1
class_weights = [ (1 / true_samples) * (len(train_results) / 2.0) , (1 / false_samples) * (len(train_results) / 2.0)]


history = model.fit(train_data, train_results, epochs=300, validation_data=(validate_data, validate_results),
                    class_weight={0: class_weights[1], 1: class_weights[0]})

# , callbacks=[save_best_model]

plt.plot(history.history["loss"])
plt.title('Model accuracy')
plt.ylabel('Cost/Total loss')
plt.xlabel('Epoch')
plt.legend(['Neural network optimization'], loc='upper right')
plt.show()

# model.load_weights(best_checkpoint_path)

model.evaluate(test_data, test_results)

model.save("best_nn_model.keras")




