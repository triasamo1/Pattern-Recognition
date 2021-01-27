import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision,Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------  Preprocess Dataset ---------

# patients_list = "IDC_regular_ps50_idx5"
#
# image_paths_class_0 = []
# image_paths_class_1 = []
# diagnosis_0 = []
# diagnosis_1 = []
# patients_0 = []
# patients_1 = []
#
# for patient in glob(patients_list + "/*"):
#     patient_id = patient.split("\\")[-1]
#     for class_ind in glob(patient + "/*"):
#         class_id = class_ind.split("\\")[-1]
#         for image_path in glob(os.path.join(class_ind, "*.png")):
#             if class_id=='0':
#                 image_paths_class_0.append(image_path)
#                 diagnosis_0.append(class_id)
#                 patients_0.append(patient_id)
#             else:
#                 image_paths_class_1.append(image_path)
#                 diagnosis_1.append(class_id)
#                 patients_1.append(patient_id)

# # labels_count = diagnosis
# # diagnosis = np.array(diagnosis)
#
# # dictionary "bible" holding all the info for each image
# bible_0 = {'patient': patients_0, 'image paths':image_paths_class_0, 'diagnosis':diagnosis_0}
# bible_1 = {'patient': patients_1, 'image paths':image_paths_class_1, 'diagnosis':diagnosis_1}
# # ========================================
# print("Class Occurences in Dataset")
# total_0=len(diagnosis_0)
# total_1=len(diagnosis_1)
# print("0 : {}".format(total_0))
# print("1 : {}".format(total_1))
# # ========================================

# # ---------- built train and test set ----------------
# X_train=[]
# Y_train=[]
# for path in bible_0['image paths'][:int(total_0 * 0.8)]:
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (50, 50))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     X_train.append(image)
#     Y_train.append('0')
#
# for path in bible_1['image paths'][:int(total_1 * 0.8)]:
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (50, 50))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     X_train.append(image)
#     Y_train.append('1')
#
# X_train = np.array(X_train)

# X_test = []
# Y_test = []
# for path in bible_0['image paths'][int(total_0 * 0.8):]:
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (50, 50))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     X_test.append(image)
#     Y_test.append('0')
#
# for path in bible_1['image paths'][int(total_1 * 0.8):]:
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (50, 50))
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     X_test.append(image)
#     Y_test.append('1')

dest_path = "breast-cancer-dataset/"

train_datagen = ImageDataGenerator(
        rescale=1./255
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
        shuffle=True,
)
test_generator = test_datagen.flow_from_directory(
        dest_path+'testing/',
        target_size=(50, 50),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# total = 0
# #total_length = len(os.listdir(dest_path+'training/0'))
# for inputs,outputs in train_generator:
#     total += 1
#     if total==158990:
#         break

# ---------  Construct Model ---------
model = Sequential([
    Conv2D(32, (3,3), input_shape=(50, 50, 3), padding='same'),
    Activation('relu'),
    Dropout(0.3),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(16,(3,3), padding='same'),
    Activation('relu'),
    Dropout(0.3),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dense(2),
    Activation('softmax')
])

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy', Precision(), Recall(), F1Score(num_classes=2)]
)

history = model.fit(
        train_generator,
        validation_data=test_generator,
        validation_steps=100,
        epochs=6,
        steps_per_epoch= 200
)

print('Training Finished..')
print('Testing ..')

# --------- Test set  ---------

probabilities = model.predict_generator(generator=test_generator)
y_true = test_generator.classes
#print(y_true)
y_pred = np.argmax(probabilities, axis=-1)


score = model.evaluate(test_generator)
print('===Testing Metrics===')
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print('Test precision: ', score[2])
print('Test recall: ', score[3])
print('Test F1 Score: ', score[4])

# ---------  Confusion Matrix ---------

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
#
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
