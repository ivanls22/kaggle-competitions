import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers.experimental import preprocessing
from matplotlib import pyplot as plt

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
label = train.loc[:, train.columns == 'label']
train_no_label = train.loc[:, train.columns != 'label']

#  Normalizing pixel darkness value
train_no_label = train_no_label / 255
test = test / 255

#  Creating an array of dataframes with each image
x_train = []
x_test = []

for idx_list in range(train.shape[0]):
    x_train.append(train_no_label.iloc[idx_list].to_numpy(dtype='float').reshape(28, 28))
x_train_dim = np.expand_dims(np.array(x_train), axis=3)

for idx_list in range(test.shape[0]):
    x_test.append(test.iloc[idx_list].to_numpy(dtype='float').reshape(28, 28))
x_test_dim = np.expand_dims(np.array(x_test), axis=3)

rotation_layer=preprocessing.RandomRotation(factor=0.25, fill_mode='constant', fill_value =0.0)
zoom_layer=preprocessing.RandomZoom(height_factor=0.25, width_factor=None, fill_mode='constant', fill_value =0.0)
trans_layer=preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode='constant', fill_value =0.0)
X_augmented = np.concatenate((x_train_dim,rotation_layer(x_train_dim), zoom_layer(x_train_dim), trans_layer(x_train_dim)))
label_augmented = np.concatenate((label, label, label, label))

print(X_augmented.shape)

for i in range(1):
    trans_layer=preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode='constant', fill_value =0.0)
    plt.imshow(trans_layer(x_train_dim[i]))
    plt.show()


early_stopping = keras.callbacks.EarlyStopping(
    min_delta=0.01,  # minimum amount of change to count as an improvement
    patience=200,  # how many epochs to wait before stopping
    restore_best_weights=True
)
#  Creating model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[28, 28, 1]),
    keras.layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(padding='same'),
    keras.layers.BatchNormalization(renorm=True),
    keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(padding='same'),
    keras.layers.BatchNormalization(renorm=True),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax')
])
#model.load_weights('my_model_weights.h5')

model.compile(
    optimizer=keras.optimizers.Adam(epsilon=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x=X_augmented,
    y=keras.utils.to_categorical(label_augmented, num_classes=10),
    shuffle=True,
    validation_split=0.20,
    epochs=1000,
    batch_size=4000,
    verbose=1,
    callbacks=[early_stopping]
)
model.save_weights('/kaggle/working/my_model_weights.h5')


epochs = history.epoch
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history_frame['val_loss'].min()))

y_predict = model.predict(x=x_test_dim)
y_result = np.argmax(y_predict, axis=1).reshape(y_predict.shape[0])
y_result_df = pd.DataFrame(columns=['ImageId', 'Label'])
y_result_df['ImageId'] = range(1, 28001)
y_result_df['Label'] = pd.Series(y_result)
y_result_df.to_csv("/kaggle/working/submision.csv", columns=["ImageId", "Label"], header=["ImageId", "Label"], index=False)

y_predict_from_training = model.predict(x=x_train_dim)
y_result_from_training = np.argmax(y_predict_from_training, axis=1).reshape(y_predict_from_training.shape[0])
cf_matrix = confusion_matrix(np.array(label), y_result_from_training)
print(cf_matrix)
print(100*sum(cf_matrix.diagonal())/42000)


model.summary()
print("Finished")
