import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Display sample image
#plt.imshow(cv2.imread("pothole_dataset/potholes/125.jpg"))
plt.show()

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

training_set = train_datagen.flow_from_directory(
    'pothole_dataset',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_set = train_datagen.flow_from_directory(
    'pothole_dataset',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Building the CNN
cnn = tf.keras.models.Sequential()

# First Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Fully Connected Layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN
cnn.fit(x=training_set, validation_data=validation_set, epochs=25)

# 모델 저장
cnn.save('pothole_detector_model.h5')

# 모델 로드
loaded_model = tf.keras.models.load_model('pothole_detector_model.h5')

# Testing the model
test_img = 'pothole_dataset/potholes/184.jpg'
test_image = image.load_img(test_img, target_size=(64, 64))

plt.imshow(cv2.imread(test_img))
plt.show()

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = loaded_model.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'pothole'
else:
    prediction = 'normal'

print(prediction)
