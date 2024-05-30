import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import roc_curve, auc

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
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)
data = data.map(lambda x, y: (x / 255.0, y))  # Normalize images

# Dataset Splitting
train_data, test_data = train_test_split(list(data), test_size=0.2, random_state=42)
X_train = np.concatenate([x for x, y in train_data])
y_train = np.concatenate([y for x, y in train_data])
X_test = np.concatenate([x for x, y in test_data])
y_test = np.concatenate([y for x, y in test_data])

# 4. Build Deep Learning Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# 5. Train with Threshold Optimization
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

class FindOptimalThreshold(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        eer_index = np.argmin(np.abs(fpr - (1 - tpr)))
        eer_threshold = thresholds[eer_index]
        print(f"\nEqual Error Rate (EER) Threshold: {eer_threshold}")
        y_pred_binary = (y_pred > eer_threshold).astype(int)
        f1 = f1_score(y_test, y_pred_binary)
        print(f"F1 Score: {f1}")

find_threshold_callback = FindOptimalThreshold()

hist = model.fit(X_train, y_train, epochs=15, validation_split=0.2, callbacks=[tensorboard_callback, find_threshold_callback])

# 6. Plot Performance
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
plt.savefig("acc");
plt.show()


# 7. Evaluate
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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

# Step 2: Calculate ROC Curve Metrics
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Step 3: Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# 8. Test
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
