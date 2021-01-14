import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import cv2
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision,Recall
from tensorflow_addons.metrics import F1Score
from keras.utils import np_utils
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------  Preprocess Dataset ---------

train_path = "fruits-360/Training"
valid_path = "fruits-360/Test"

fruit_images = []
labels = []
for fruit_dir_path in glob(train_path + "/*"):
    fruit_label = fruit_dir_path.split("\\")[-1]
    for image_path in glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (100, 100))               # <-- We resize images at 50x50
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels_count = labels
labels = np.array(labels)

# ========================================
print("Class Occurences in Dataset")
for lab in np.unique(labels):
    print("{0} : {1} ".format(lab,labels_count.count(lab)))

# ========================================

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

        image = cv2.resize(image, (100, 100))             # <-- We resize images at 50x50
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)

validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])

X_train, X_test = fruit_images, validation_fruit_images
Y_train, Y_test = label_ids, validation_label_ids

# Normalize color values to between 0 and 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

# OneHot Encode the Output
Y_train = np_utils.to_categorical(Y_train, 8)
Y_test = np_utils.to_categorical(Y_test, 8)

print("Train Set Shape: ", X_train.shape)
print("Test Set Shape: ", X_test.shape)

# ---------  Construct Model ---------
model = Sequential([
    Conv2D(32, (3,3), input_shape=(100, 100, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(16,(3,3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(8,(3,3), padding='same'),
    GlobalAveragePooling2D(),
    Activation('softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=8)]
)

history = model.fit(X_train, Y_train, batch_size=128, epochs=14, shuffle=True)

print('Training Finished..')
print('Testing ..')

# --------- Test set  ---------

score = model.evaluate(X_test, Y_test)

print('===Testing Metrics===')
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print('Test precision: ', score[2])
print('Test recall: ', score[3])
print('Test F1 Score: ', score[4])

# ---------  Confusion Matrix ---------

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

conf_mat = confusion_matrix(np.argmax(Y_test, axis=-1), y_pred)
f,ax=plt.subplots(figsize=(5,5))
# Normalize the confusion matrix.
conf_mat = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
plt.title("Confusion matrix")
sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
tick_marks = np.arange(len(label_to_id_dict.keys()))
plt.xticks(tick_marks, label_to_id_dict.keys(), rotation=45)
plt.yticks(tick_marks, label_to_id_dict.keys(), rotation=45)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# ---------  Accuracy - Loss Plot ---------
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
