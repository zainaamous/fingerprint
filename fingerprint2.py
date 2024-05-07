import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""# 2. Remove Dodgy Images"""

data_dir = 'comp432'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            with Image.open(image_path) as img:
                img_format = img.format.lower()
                if img_format not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

"""# 3. Load Data"""

data = tf.keras.utils.image_dataset_from_directory('comp432')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Assuming 'data' contains features and labels for both genuine users and imposters
X = []
y = []
for images, labels in data:
    for img in images:
        X.append(img)
    for lbl in labels:
        y.append(lbl)
X = np.array(X)
y = np.array(y)

# Dataset Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# 4. Scale Data"""

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

"""# 5. Build Deep Learning Model"""

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#


# Additional modifications and training options can be explored


"""# 6. Train"""

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(X_train_scaled, y_train, epochs=15, validation_split=0.2, callbacks=[tensorboard_callback])
# hist = model.fit(data_augmentation(X_train_scaled), y_train, epochs=20, validation_split=0.2, callbacks=[tensorboard_callback])
"""# 7. Plot Performance"""

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

"""# 8. Evaluate"""

y_pred = model.predict(X_test_scaled)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate False Match Rate (FMR) and False Non-Match Rate (FNMR)
fmr = fpr  # FMR is the same as False Positive Rate
fnmr = 1 - tpr  # FNMR is 1 - True Positive Rate
print("FMR:",fmr)
print("FNMR:", fnmr)

# Plot the Receiver Operating Characteristic (ROC) curve
plt.plot(fmr, fnmr, color='teal', label='ROC curve')
plt.xlabel('False Match Rate (FMR)')
plt.ylabel('Flase Non Match Rate (FNMR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Determine the Equal Error Rate (EER)
eer_index = np.argmin(np.abs(fmr - fnmr))
eer_threshold = thresholds[eer_index]
eer = (fmr[eer_index] + fnmr[eer_index]) / 2
print("Equal Error Rate (EER):", eer)

"""# 9. Test"""

img = cv2.imread('314.jpg')
resize = cv2.resize(img, (256, 256))
resize_scaled = resize / 255
yhat = model.predict(np.expand_dims(resize_scaled, 0))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(resize)
plt.show()

if yhat > 0.65:
    print(f'Predicted class is imposter')
else:
    print(f'Predicted class is genuine')
