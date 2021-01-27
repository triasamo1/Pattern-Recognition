import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import Precision,Recall
from tensorflow_addons.metrics import F1Score
from  tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dest_path = "breast-cancer-dataset/"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        # rotation_range=90,
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
        dest_path+'training/',
        target_size=(50, 50),
        batch_size=64,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
)
test_generator = test_datagen.flow_from_directory(
        dest_path+'testing',
        target_size=(50, 50),
        batch_size=64,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False)

# ---------  Construct Model ---------
# model = Sequential([
#     ResNet50( weights="imagenet", input_shape=(50, 50, 3), include_top=False),
#     Flatten(),
#     Dense(1),
#     Activation('sigmoid')
# ])


model = Sequential([
    ResNet50( weights=None, input_shape=(50, 50, 3), include_top=True, classes=2),
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=2)]
)
#add momentum,
history = model.fit(
        train_generator,
        epochs=6,
        steps_per_epoch=200,
        validation_data= test_generator,
        validation_steps=100
)

print('Training Finished..')
print('Testing ..')

# --------- Test set  ---------

# score = model.evaluate(test_generator)
# print('===Testing Metrics===')
# print('Test loss: ', score[0])
# print('Test accuracy: ', score[1])
# print('Test precision: ', score[2])
# print('Test recall: ', score[3])
# print('Test F1 Score: ', score[4])
# ---------  Confusion Matrix ---------

probabilities = model.predict_generator(generator=test_generator)
y_true = test_generator.classes
y_pred = np.argmax(probabilities, axis=-1)

conf_mat = confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(5,5))
# Normalize the confusion matrix.
conf_mat = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)
plt.title("Confusion matrix")
sns.heatmap(conf_mat,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

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

