import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


def main():
    # data prepare
    trainDir = r'{0}'.format(os.getcwd()+'\\training_set\\training_set')

    trainData = tf.keras.utils.image_dataset_from_directory(
        trainDir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=1,
        validation_split=0.1,
        subset='training',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    testDir = r'{0}'.format(os.getcwd()+'\\test_set\\test_set')

    testData = tf.keras.utils.image_dataset_from_directory(
        testDir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=1,
        validation_split=0.1,
        subset='validation',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,

    )
    # create structure cnn model
    model = tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(256, 256, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    # info model
    model.summary()
    # compile
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # train
    history = model.fit(
        trainData,
        epochs=10,
        validation_data=testData,
    )
    predictImage = tf.keras.utils.load_img(r'{0}'.format(
        os.getcwd()+'\\predict_img.jpg'), target_size=(256, 256))
    plt.imshow(predictImage)
    predictImage = tf.keras.utils.img_to_array(predictImage)
    predictImage = np.expand_dims(predictImage, axis=0)
    result = model.predict(predictImage)
    if (result >= 0.5):
        
        plt.annotate('Dog', xy=(128, 128), xycoords='data',
                     xytext=(0.1, 0.5), textcoords='figure fraction',
                     arrowprops=dict(arrowstyle="->"))
    else:
        plt.annotate('Cat', xy=(128, 128), xycoords='data',
                     xytext=(0.1,0.5), textcoords='figure fraction',
                     arrowprops=dict(arrowstyle="->"))
    plt.scatter(128, 128, s=500, c='red', marker='o')
    plt.show()


if __name__ == "__main__":
    main()
