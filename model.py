import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

my_batch_size = 128
my_epochs = 20
epochs=20
my_steps_per_epoch = 100
my_validation_steps = 128


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    #tf.keras.layers.Flatten(input_shape=(28,28,3)),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Dense(10)
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#model.compile(optimizer='adam',
#              loss=loss_fn,
#              metrics=['accuracy'])
print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=my_batch_size,
                                                    class_mode='binary',
                                                    target_size=(28, 28))

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=my_batch_size,
                                                              class_mode='binary',
                                                              target_size=(28, 28))
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test,  y_test, verbose=2)

#probability_model = tf.keras.Sequential([
#  model,
#  tf.keras.layers.Softmax()
#])

#probability_model(x_test[:5])




history = model.fit(train_generator,
                              epochs=my_epochs,
                              validation_data=validation_generator,
                              batch_size = my_batch_size,
                              steps_per_epoch = my_steps_per_epoch,
                              validation_steps = my_validation_steps)

