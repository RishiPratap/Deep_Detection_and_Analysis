import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Define data directory and parameters
data_dir = 'data'
image_size = (128, 128)
batch_size = 64
epochs = 30

# Load and preprocess the dataset
def load_and_preprocess_data(data_dir, image_size):
    real_images = []
    fake_images = []

    # Load real images
    for filename in os.listdir(os.path.join(data_dir, 'real')):
        img = Image.open(os.path.join(data_dir, 'real', filename))
        img = img.resize(image_size)
        real_images.append(np.array(img))

    # Load fake images
    for filename in os.listdir(os.path.join(data_dir, 'fake')):
        img = Image.open(os.path.join(data_dir, 'fake', filename))
        img = img.resize(image_size)
        fake_images.append(np.array(img))

    # Create labels (1 for real, 0 for fake)
    real_labels = np.ones(len(real_images))
    fake_labels = np.zeros(len(fake_images))

    # Concatenate and shuffle the data
    all_images = np.array(real_images + fake_images)
    all_labels = np.concatenate([real_labels, fake_labels])
    indices = np.arange(len(all_images))
    np.random.shuffle(indices)
    all_images = all_images[indices]
    all_labels = all_labels[indices]

    return all_images, all_labels

# Load and preprocess the data
data, labels = load_and_preprocess_data(data_dir, image_size)

# Split the data into training, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=42)

# Build and compile the model with architectural changes
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

# Compile the model with a lower initial learning rate
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_labels))

# Extract training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot training history for accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training history for loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Save the figures
plt.savefig('training_history.png')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test accuracy:", test_acc)

# Generate classification report
y_pred = (model.predict(test_data) > 0.5).astype("int32")
print(classification_report(test_labels, y_pred))

# Save the trained model
model.save('model/deep_fake_model_tf.h5')