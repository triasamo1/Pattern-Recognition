import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import cv2
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Precision,Recall
from tensorflow_addons.metrics import F1Score
from keras.utils import np_utils
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dest_path = "breast-cancer-dataset/"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
        dest_path+'training/',
        target_size=(50, 50),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
)
test_generator = test_datagen.flow_from_directory(
        dest_path+'testing',
        target_size=(50, 50),
        batch_size=32,
        class_mode='categorical')

# ---------  Construct Model ---------
resnet = ResNet50( weights="imagenet", input_shape=(50, 50, 3), include_top=False, classes=2 )

model = Sequential([
    resnet(trainable=False),
    Flatten()(resnet.output),
    Dense(2),
    Activation('softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=2)]
)

history = model.fit(
        train_generator,
        epochs=14,
        steps_per_epoch=200
)

print('Training Finished..')
print('Testing ..')

# --------- Test set  ---------

score = model.evaluate(test_generator)

print('===Testing Metrics===')
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print('Test precision: ', score[2])
print('Test recall: ', score[3])
print('Test F1 Score: ', score[4])
# ---------  Confusion Matrix ---------

# y_pred = model.predict(X_test)
# y_pred = np.argmax(y_pred, axis=-1)
#
# conf_mat = confusion_matrix(np.argmax(Y_test, axis=-1), y_pred)
# f,ax=plt.subplots(figsize=(5,5))
# # Normalize the confusion matrix.
# conf_mat = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
# plt.title("Confusion matrix")
# sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
# tick_marks = np.arange(len(label_to_id_dict.keys()))
# plt.xticks(tick_marks, label_to_id_dict.keys(), rotation=45)
# plt.yticks(tick_marks, label_to_id_dict.keys(), rotation=45)
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# ---------  Accuracy - Loss Plot ---------
fit_hist = pd.DataFrame(history.history)

loss = round(np.min(fit_hist['loss']), 2)
val_loss = round(np.min(fit_hist['val_loss']), 2)

plt.title(f"Train Loss ({loss}) and Test Loss ({val_loss})")
plt.plot(fit_hist['loss'], label='Train Loss')
plt.plot(fit_hist['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(color='#e6e6e6')
plt.legend()
plt.show()

acc = round(np.max(fit_hist['accuracy']), 2)
val_acc = round(np.max(fit_hist['val_accuracy']), 2)

plt.title(f"Train Accuracy ({acc}) and Test Accuracy ({val_acc})")
plt.plot(fit_hist['accuracy'], label='Train Accuracy')
plt.plot(fit_hist['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(color='#e6e6e6')
plt.legend()
plt.show()

