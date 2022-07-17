import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

x_tr = np.load("dataml200/ex1/teenmagi2022/training_x.dat", allow_pickle=True)
y_tr = np.load("dataml200/ex1/teenmagi2022/training_y.dat", allow_pickle=True)
x_val = np.load("dataml200/ex1/teenmagi2022/validation_x.dat", allow_pickle=True)

# Reshape data
x_train = []
for data in x_tr:
    x_train.append(data[:, :, 0].reshape(64))

train_X = np.array(x_train).reshape(-1, 1, 8, 8).transpose(0, 2, 3, 1).astype('uint8')
x_train = None
val_X = np.array(x_val)[:, :, :, 0].reshape(-1, 1, 8, 8).transpose(0, 2, 3, 1).astype('uint8')
# -1 because labels go from 1-1000
train_Y = np.array(y_tr).astype(int) - 1

# Normalize
#train_X = train_X/255.0
#val_X = val_X/255.0

# Scale data
train_X = train_X.reshape(-1, 64)
val_X = val_X.reshape(-1, 64)

scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
val_X = scaler.transform(val_X)

train_X = train_X.reshape(-1, 8, 8, 1)
val_X = val_X.reshape(-1, 8, 8, 1)

'''
# Some test data for accuracy calculations
n = 2
hop_size = 24*n
x_test = np.array(train_X[::hop_size])
y_test = np.array(train_Y[::hop_size])

test_inds = np.arange(0, len(train_X), hop_size)

# Remove test samples from training samples
train_X = np.delete(train_X, test_inds, axis=0)
train_y = np.delete(train_Y, test_inds, axis=0)
'''

# Mirror and flip imgs
#X_mirrored = train_X[...,:,::-1].copy()[::2]
#X_flipped = train_X[...,::-1,:].copy()
#X_mirror_flipped = train_X[...,::-1,::-1].copy()

#train_X = np.concatenate((train_X, X_mirrored))
#train_Y = np.concatenate((train_Y, train_Y[::2]))

# Split data to training and validation samples
train_X, validation_X, train_Y, validation_y = train_test_split(train_X, train_Y, test_size=0.33, random_state=1)

'''
train_X = np.array(train_X[::12])
train_Y = np.array(train_Y[::12])

validation_X = np.array(validation_X[::12])
validation_y= np.array(validation_y[::12])

# Mirror and flip imgs
X_mirrored = train_X[...,:,::-1].copy()[::2]
#X_flipped = train_X[...,::-1,:].copy()
#X_mirror_flipped = train_X[...,::-1,::-1].copy()

train_X = np.concatenate((train_X, X_mirrored))
train_Y = np.concatenate((train_Y, train_Y[::2]))

X_val_mirrored = validation_X[...,:,::-1].copy()[::2]
#X_flipped = train_X[...,::-1,:].copy()
#X_mirror_flipped = train_X[...,::-1,::-1].copy()

validation_X = np.concatenate((validation_X, X_val_mirrored))
validation_y = np.concatenate((validation_y, validation_y[::2]))
'''

# Limit data
#train_X = train_X[np.mod(np.arange(len(train_X)), n) != 0][::3]
#train_Y = train_Y[np.mod(np.arange(len(train_Y)), n) != 0][::3]
#print(len(train_X))

# Transfer labels to one-hot vectors
oh_train_Y = np.zeros((train_Y.size, train_Y.max() + 1))
oh_train_Y[np.arange(train_Y.size), train_Y] = 1

oh_val_Y = np.zeros((validation_y.size, validation_y.max() + 1))
oh_val_Y[np.arange(validation_y.size), validation_y] = 1

'''
model = keras.models.load_model("dataml200/ex1/model.h5")

# Prediction on validation data
validation_prediction = model.predict(val_X)
valid_preds = np.argmax(validation_prediction, axis=1) + 1

# Create Ids
ids = np.arange(start=1, stop=len(valid_preds) + 1, step=1)

# Write CSV file
df = pd.DataFrame({"Id": ids, "Class": valid_preds})
df.to_csv("dataml200/ex1/predictionsNN.csv", index=False)
'''

'''
model = keras.models.load_model("dataml200/ex1/model.h5")
checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'dataml200/ex1/model.h5', save_best_only=True,
                                                 monitor='val_acc', mode='max', period=1, verbose=1)

model.fit(train_X, oh_train_Y, validation_data=(validation_X, oh_val_Y), batch_size=512, epochs=20,
          verbose=1, callbacks=[checkpoint])


model.save(f"dataml200/ex1/1model.h5")
'''


''''''
for i in range(5):

    # CNN layers
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(512, (2, 2), activation='relu', input_shape=(8, 8, 1)))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512, (2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512, (2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512, (2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Dropout(0.25))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    # 1000 output
    model.add(keras.layers.Dense(1000, activation='softmax'))

    # Compile and train
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=f'dataml200/ex1/model.h5', save_best_only=True,
                                                 monitor='val_acc', mode='max', period=1, verbose=1)

    model.fit(train_X, oh_train_Y, validation_data=(validation_X, oh_val_Y), batch_size=512, epochs=25,
              verbose=1, callbacks=[checkpoint])

    model.save(f"dataml200/ex1/model{i}.h5")
    
    break
    


'''
model = keras.models.load_model(f"dataml200/ex1/model.h5")
prediction = model.predict(x_test)

# One-hot to integer vector, +1 to correct labels
preds = np.argmax(prediction, axis=1) + 1

corr_pred = len(np.where(y_test-preds == 0)[0])
acc = corr_pred/len(y_test)
print(f"Accuracy for testing data: {acc}")

model.save(f"dataml200/ex1/{acc:.5f}model.h5")
'''



model = keras.models.load_model("dataml200/ex1/model1.h5")

# Prediction on validation data
validation_prediction = model.predict(val_X)
valid_preds = np.argmax(validation_prediction, axis=1) + 1

# Create Ids
ids = np.arange(start=1, stop=len(valid_preds) + 1, step=1)

# Write CSV file
df = pd.DataFrame({"Id": ids, "Class": valid_preds})
df.to_csv("dataml200/ex1/predictionsNN.csv", index=False)
''''''


'''

model = keras.models.load_model("dataml200/ex1/0.00446modelOverfit.h5")

model.fit(train_X, oh_train_Y, validation_data=(validation_X, oh_val_Y), batch_size=1024, epochs=30,
          verbose=1)

prediction = model.predict(x_test)

# One-hot to integer vector, +1 to correct labels
preds = np.argmax(prediction, axis=1) + 1

corr_pred = len(np.where(y_test-preds == 0)[0])
acc = corr_pred/len(y_test)
print(f"Accuracy for testing data: {acc}")

model.save(f"dataml200/ex1/{acc:.5f}modelOverfit.h5")
'''
