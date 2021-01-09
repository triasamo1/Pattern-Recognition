# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import cv2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision,Recall
from tensorflow.keras.preprocessing import image
from tensorflow_addons.metrics import F1Score
from keras.utils import np_utils
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
# TODO: add callbacks to stop training at certain achievement
# TODO: L2 Regularization to stop overfitting


# training and testing path
train_path = "fruits-360/Training"
valid_path = "fruits-360/Test"

fruit_images = []
labels = []
for fruit_dir_path in glob(train_path + "/*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (50, 50))               # <-- We resize images at 50x50
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])

print(fruit_images.shape, label_ids.shape, labels.shape)

validation_fruit_images = []
validation_labels = []
for fruit_dir_path in glob(valid_path + "/*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (50, 50))             # <-- We resize images at 50x50
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)

validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])

X_train, X_test = fruit_images, validation_fruit_images
Y_train, Y_test = label_ids, validation_label_ids

# Normalize color values to between -1 and 1
X_train = np.subtract(X_train/127.5, 1)
X_test = np.subtract(X_test/127.5, 1)

# Make a flattened version for some of our models
X_train = X_train.reshape(X_train.shape[0], 50*50*3)
X_test = X_test.reshape(X_test.shape[0], 50*50*3)

# One Hot Encode the Output
Y_train = np_utils.to_categorical(Y_train, 8)
Y_test = np_utils.to_categorical(Y_test, 8)
print(X_train.shape)

# construct model
model = Sequential([
    Dense(64, input_shape=(7500,)),            # <-- Make our network smaller/simpler
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(8),
    Activation('softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=8)]
)

history = model.fit(X_train, Y_train, batch_size=64, epochs=4)

print('Training Finished..')
print('Testing ..')

score = model.evaluate(X_test, Y_test)

print('===Testing Metrics===')
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print('Test precision: ', score[2])
print('Test recall: ', score[3])
print('Test F1 Score: ', score[4])

y_pred = model.predict(X_train)
y_pred = np.argmax(y_pred, axis=-1)

conf_mat = confusion_matrix(np.argmax(Y_train, axis=-1), y_pred)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.show()

fit_hist = pd.DataFrame(history.history)

loss = round(np.min(fit_hist['loss']), 2)
acc = round(np.max(fit_hist['accuracy']), 2)

plt.title(f"Train Loss ({loss}) and Train Accuracy ({acc})")
plt.plot(fit_hist['loss'], label='Train Loss')
plt.plot(fit_hist['accuracy'], label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss & Accuracy')
plt.grid(color='#e6e6e6')
plt.legend()
plt.show()
