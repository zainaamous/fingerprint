import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import precision_recall_curve

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""# 2. Remove corrubted Images"""

data_dir = 'comp432'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
# this goes thru all images and remove images with forbidden format
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
# to build data pipeline, so rather thanloading everything into memory, which allows you to scale to much larger
# it also gives you a repeatable set of steps
# A data pipeline is a series of steps or processes that are used to ingest, process, transform, and analyze data.
# It's essentially a workflow that facilitates the movement of data from its source to its destination in a structured and efficient manner
data = tf.keras.utils.image_dataset_from_directory('comp432')
# allows us to access the data from our data pipeline
# it automatichally reshapes our images to make sure they have a consistent size.
data_iterator = data.as_numpy_iterator()
# get us a batch back (the data)
# actually acessing the data pipeline

# NOTE: there are two parts of a dataset, the images themself and their labels.

# so batch holds our data (images) as numpy arrays
# access to our iterator and next is going ro grab our next batch
batch = data_iterator.next()
# we got 32 images of shape 256X256 by three channels these color channels "rgb" each has a value ranging from 0-255

# now when building AI models we want our values to be as small as possible so our model can generate faster

# and the optimal way to do is by scaling the data
batch[0].shape


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

# scaling data by dividing value by 255 so instead of our images values ranging from 0-255, they range from 0 to 1.
"""# 4. Scale Data"""

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

"""# 5. Build Deep Learning Model"""
# Sequential API for simpler, linear models with a single input and a single output, where layers are stacked sequentially
#
# Functional API for more complex models with multiple inputs/outputs
# or with intricate architectures like residual connections or shared layers.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)), #input layer is our first layer
    # relu is going to take the output of that conv layer and pass it to a function that is going to convert outputs beloqw zero to zero
    # and its going to preserve positive values
    # this helps us taking into account non-linear patterns


    # # 16 filters, each filter has 3X3 pixles and filters are gonna move one pixle at a time
    tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'),
    # MaxPooling layer  takes the maximum value after the relu activation and its going to turn back that value
    # its effectivly going to condense that inforrmation - reduce image data
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # after applying conv layers all those filters are going to condense the rows
    # and the width and then the number of filters will form the channel values
    # get a single value
    tf.keras.layers.Flatten(),

    # fully connected layers
    # 256 nuerons
    tf.keras.layers.Dense(256, activation='relu'),

    # final layer is a single dense layer = single output which is zero or one
    # sigmoid function return a value of zero or one  which in our case is going to map to a genuine or imposter user
    tf.keras.layers.Dense(1, activation='sigmoid')
])


'''
three conv layers, two dense layer

dense layer serves to transform the spatially-organized features extracted by the convolutional layers
into a flat vector format suitable for classification or regression tasks.
It applies a matrix multiplication operation followed by a non-linear activation function to produce the final output of the network.
'''
# NOTE: we are taking all of the outputs from a layer and we are passing it to an activation function to modify what the output looks like



# adam is an optimizer
# so  far its binary classification so we used 'BinaryCrossentropy'
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


'''

optimizers are responsible for adjusting the parameters (weights and biases)
of the network during training to minimize the loss function. 
The goal of optimization is to find the optimal set of parameters that minimize the difference between the predicted output of the model and the actual target output.
'''

"""# 6. Train with Threshold Optimization"""
# # just a var that points to logs
logdir = 'logs'
# # save model at a specific checkpoint
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Callback to find optimal threshold during training
# Define a custom callback class named FindOptimalThreshold, inheriting from tf.keras.callbacks.Callback.
class FindOptimalThreshold(tf.keras.callbacks.Callback):
    # Define a method that triggers at the end of each epoch during training.
    def on_epoch_end(self, epoch, logs=None):
        # Predict probabilities of positive class for the test dataset using the model.
        y_pred = self.model.predict(X_test_scaled)

        # Compute the Receiver Operating Characteristic (ROC) curve.
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        # Find the threshold where False Positive Rate (FPR) is closest to (1 - True Positive Rate) for Equal Error Rate (EER).
        eer_index = np.argmin(np.abs(fpr - (1 - tpr)))  # EER index
        eer_threshold = thresholds[eer_index]  # Threshold corresponding to EER

        # Print the Equal Error Rate (EER) Threshold.
        print("\nEqual Error Rate (EER) Threshold:", eer_threshold)

        # Adjust predictions using the EER threshold to classify as positive or negative.
        y_pred_binary = (y_pred > eer_threshold).astype(int)

        # Evaluate the F1 score, which balances precision and recall.
        f1 = f1_score(y_test, y_pred_binary)
        # Print the F1 score.
        print("F1 Score:", f1)

# Instantiate the FindOptimalThreshold callback.
find_threshold_callback = FindOptimalThreshold()


# Train the model with threshold optimization callback

# epochs is how long are we training for, one epoch is one run over our entire training set of data
# then run evaluation on our validation data so we can see how well our model perform in real time
hist = model.fit(X_train_scaled, y_train, epochs=15, validation_split=0.2, callbacks=[tensorboard_callback, find_threshold_callback])

"""# 7. Plot Performance"""
# plotting the loss
# loss should go down as we train
# Note: if 'Loss' started going up that means that we are overfitting.
# Solution to that is regularization.
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
# accuracy should go up as we train
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
print("FMR:", fmr)
print("FNMR:", fnmr)

# Plot the Receiver Operating Characteristic (ROC) curve
plt.plot(fmr, tpr, color='teal', label='ROC curve')
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

