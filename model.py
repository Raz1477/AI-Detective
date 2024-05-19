# Import Statements
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data File Locations
train_data_dir = 'Data/train/'
test_data_dir = 'Data/test/'

# Data Parameters
batch_size = 32
img_height = 32
img_width = 32

# Training Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=True,
)

# Validation Dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=True,
)

# Test Dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    shuffle=False
)

# Class Names
class_names=train_ds.class_names

# Plotting First 9 Images
'''
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        label = labels[i].numpy().astype(int)
        plt.title(class_names[label])
        plt.axis('off')
'''

# Dataset Optimization
AUTOTONE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTONE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTONE)

# Training Model
num_classes = len(class_names)

# Defining Model
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.experimental.preprocessing.Resizing(32, 32),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(num_classes)
])

# Compiling Model
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
  )

# Fitting the Model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=9
)

# Testing the Model
model.evaluate(test_ds)
print(model.summary())
predictions = model.predict(test_ds)

# Accuracy and Loss Variables
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Graphing validation and training loss vs. # of epochs
fig, ax1 = plt.subplots()
ax1.plot(epochs, loss, 'r', label='Training Loss')
ax1.plot(epochs, val_loss, 'b', label='Validation Loss')
ax1.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation Loss')
ax1.legend()

# Graphing validation and training accuracy vs. # of epochs
fig1, ax2 = plt.subplots()
ax2.plot(epochs, acc, 'r', label='Training Accuracy')
ax2.plot(epochs, val_acc, 'b', label='Validation Accuracy')
ax2.set(xlabel='Epochs', ylabel='Accuracy', title='Training and Validation Accuracy')
ax2.legend()

plt.show()