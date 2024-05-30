# import tensorflow as tf
# import os
# import cv2
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, f1_score
#
# # Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#
# # 2. Remove corrupted Images
# data_dir = 'comp432'
# image_exts = ['jpeg', 'jpg', 'bmp', 'png']
#
# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         try:
#             with Image.open(image_path) as img:
#                 img_format = img.format.lower()
#                 if img_format not in image_exts:
#                     print(f'Image not in ext list {image_path}')
#                     os.remove(image_path)
#         except Exception as e:
#             print(f'Issue with image {image_path}: {e}')
#
# # 3. Load Data
# data = tf.keras.utils.image_dataset_from_directory('comp432')
# data_iterator = data.as_numpy_iterator()
# batch = data_iterator.next()
# print(batch[0].shape)
#
# # Assuming 'data' contains features and labels for both genuine users and imposters
# X = []
# y = []
# for images, labels in data:
#     X.extend(images)
#     y.extend(labels)
# X = np.array(X)
# y = np.array(y)
#
# # Dataset Splitting
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 4. Scale Data
# X_train_scaled = X_train / 255.0
# X_test_scaled = X_test / 255.0
#
# # 5. Build Deep Learning Model
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(256, 256, 3)),
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
#
# # 6. Train with Threshold Optimization
# logdir = 'logs'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#
# class FindOptimalThreshold(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         y_pred = self.model.predict(X_test_scaled)
#         fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#         eer_index = np.argmin(np.abs(fpr - (1 - tpr)))
#         eer_threshold = thresholds[eer_index]
#         print(f"\nEqual Error Rate (EER) Threshold: {eer_threshold}")
#         y_pred_binary = (y_pred > eer_threshold).astype(int)
#         f1 = f1_score(y_test, y_pred_binary)
#         print(f"F1 Score: {f1}")
#
# find_threshold_callback = FindOptimalThreshold()
#
# hist = model.fit(X_train_scaled, y_train, epochs=15, validation_split=0.2, callbacks=[tensorboard_callback, find_threshold_callback])
#
# # 7. Plot Performance
# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='loss')
# plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
# fig.suptitle('Loss', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()
#
# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
# plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
# fig.suptitle('Accuracy', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()
#
# # 8. Evaluate
# y_pred = model.predict(X_test_scaled)
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# fmr = fpr
# fnmr = 1 - tpr
#
# eer_index = np.argmin(np.abs(fmr - fnmr))
# eer_threshold = thresholds[eer_index]
# eer = (fmr[eer_index] + fnmr[eer_index]) / 2
# print(f"Equal Error Rate (EER): {eer}")
#
# # EER Plot
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, fmr, label='FAR (False Accept Rate)', color='blue')
# plt.plot(thresholds, fnmr, label='FRR (False Reject Rate)', color='red')
#
# plt.plot(thresholds[eer_index], eer, 'ro')
# plt.text(thresholds[eer_index], eer, f'EER\n({eer:.2f})', ha='right', va='bottom', color='red')
#
# plt.xlabel('Threshold')
# plt.ylabel('Rate')
# plt.title('EER Graph')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()
#
# # 9. Test
# img = cv2.imread('314.jpg')
# resize = cv2.resize(img, (256, 256))
# resize_scaled = resize / 255.0
# yhat = model.predict(np.expand_dims(resize_scaled, 0))
#
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
#
# plt.imshow(resize)
# plt.show()
#
# if yhat > eer_threshold:
#     print('Predicted class is imposter')
# else:
#     print('Predicted class is genuine')
#
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 2. Remove corrupted Images
data_dir = 'comp432'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            with Image.open(image_path) as img:
                img_format = img.format.lower()
                if img_format not in image_exts:
                    print(f'Image not in ext list {image_path}')
                    os.remove(image_path)
        except Exception as e:
            print(f'Issue with image {image_path}: {e}')

# 3. Load Data
data = tf.keras.utils.image_dataset_from_directory(
    'comp432',
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary'
)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Apply data augmentation to training data
augmented_data = data.map(lambda x, y: (data_augmentation(x, training=True), y))

# Split dataset into training and validation sets
train_data = augmented_data.take(int(len(augmented_data) * 0.8))
val_data = augmented_data.skip(int(len(augmented_data) * 0.8))

# Prefetch data for better performance
train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 4. Scale Data (handled automatically by Keras)
normalization_layer = tf.keras.layers.Rescaling(1./255)

# 5. Build Deep Learning Model
model = tf.keras.Sequential([
    normalization_layer,
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train with Threshold Optimization
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

class FindOptimalThreshold(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(val_data)
        y_true = np.concatenate([y for x, y in val_data], axis=0)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        eer_index = np.argmin(np.abs(fpr - (1 - tpr)))
        eer_threshold = thresholds[eer_index]
        print(f"\nEqual Error Rate (EER) Threshold: {eer_threshold}")
        y_pred_binary = (y_pred > eer_threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        print(f"F1 Score: {f1}")

find_threshold_callback = FindOptimalThreshold()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

hist = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[tensorboard_callback, find_threshold_callback, early_stopping]
)

# 7. Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# 8. Evaluate
y_pred = model.predict(val_data)
y_true = np.concatenate([y for x, y in val_data], axis=0)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
fmr = fpr
fnmr = 1 - tpr

eer_index = np.argmin(np.abs(fmr - fnmr))
eer_threshold = thresholds[eer_index]
eer = (fmr[eer_index] + fnmr[eer_index]) / 2
print(f"Equal Error Rate (EER): {eer}")

# EER Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, fmr, label='FAR (False Accept Rate)', color='blue')
plt.plot(thresholds, fnmr, label='FRR (False Reject Rate)', color='red')

plt.plot(thresholds[eer_index], eer, 'ro')
plt.text(thresholds[eer_index], eer, f'EER\n({eer:.2f})', ha='right', va='bottom', color='red')

plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('EER Graph')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# 9. Test
img = cv2.imread('314.jpg')
resize = cv2.resize(img, (256, 256))
resize_scaled = resize / 255.0
yhat = model.predict(np.expand_dims(resize_scaled, 0))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(resize)
plt.show()

if yhat > eer_threshold:
    print('Predicted class is imposter')
else:
    print('Predicted class is genuine')
