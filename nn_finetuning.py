import os
import cv2
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from joblib import dump, load

random.seed(50)
BATCH_SIZE = 100
CHANNEL = 3
CLASSES = 1

#### Function to move pictures to their corresponding folders based on labels
def move_pics():
    old_path = "ISIC_2019_Training_Input/"
    labs = pd.read_csv("ISIC_2019_Training_GroundTruth.csv")
    labs_img = labs['image']  + '.jpg'
    y = labs.drop(['image'], axis = 1).idxmax(axis=1).ravel()
    y[(y == 'NV') | (y == 'AK') | (y == 'BKL') | (y == 'DF') | (y == 'VASC')] = 0
    y[(y == 'MEL') | (y == 'SCC') | (y == 'BCC')] = 1
    y=y.astype('int')
    labs = pd.concat([labs_img,pd.Series(y)], axis=1)
    labs = labs.rename({0:"diagnosis"}, axis='columns')
    benign_path = "input/benign/"
    malignant_path = "input/malignant/"
    os.mkdir("input")
    os.mkdir("input/benign")
    os.mkdir("input/malignant")
    for i in range(0,labs.shape[0]):
        if labs['diagnosis'][i] == 1:
            os.rename(old_path + labs['image'][i], malignant_path + labs['image'][i])
        else: 
            os.rename(old_path + labs['image'][i], benign_path + labs['image'][i])

class NeuralNetwork_Performance:
    def __init__(self, preprocess, model, callbacks, RESIZED):
        self.preprocess = preprocess
        self.base_model = model
        self.model = None
        self.RESIZED = RESIZED
   
    def import_data(self, train_size, val_size, test_size):
        print("data importing")
        traintest_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "input",
            seed=422,
            image_size=(self.RESIZED, self.RESIZED))
        train_ds = traintest_ds.take(train_size)
        test_ds = traintest_ds.skip(train_size)
        val_ds = traintest_ds.skip(test_size)
        test_ds = traintest_ds.take(test_size)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                "input", seed = 422,
                shuffle=True,
                batch_size=BATCH_SIZE,
                image_size=(RESIZED, RESIZED),
                validation_split=0.4,
                subset='training')
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                "input", seed = 422,
                shuffle=True,
                batch_size=BATCH_SIZE,
                image_size=(RESIZED, RESIZED),
                validation_split=0.4,
                subset='validation')
        val_batches = tf.data.experimental.cardinality(val_ds)
        test_ds = val_ds.take(val_batches // 2)
        val_ds = val_ds.skip(val_batches // 2)

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
        return(train_ds, val_ds, test_ds)

    def data_preprocessing(self):
        print("data processing")
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
            ])
        inputs = tf.keras.Input(shape=(self.RESIZED, self.RESIZED) + (CHANNEL,))
        x = data_augmentation(inputs)
        x = self.preprocess(x)
        return(x, inputs)

    def model_first_step_compile(self, x, inputs, train_ds):
        print("Compiling model")
        self.base_model.trainable = False
        image_batch, label_batch = next(iter(train_ds))
        feature_batch = self.base_model(image_batch)
        global_average_layer = tf.keras.layers.GlobalMaxPool2D()
        feature_batch_average = global_average_layer(feature_batch)
        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        x = self.base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer= keras.optimizers.SGD(lr=0.01, momentum=0.5, decay=0.005),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
       #### saving the model
        return(self.model)

    def model_second_step_compile(self, x, inputs, train_ds):
        for layer in self.base_model.layers:
                if "BatchNormalization" not in layer.__class__.__name__:
                    layer.trainable = True
        self.model.compile(optimizer= keras.optimizers.SGD(lr=0.001, momentum=0.5, decay=0.001),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        return(self.model)

    def model_eval(self, train_ds, val_ds, callbacks, name):
        print("evaluating model")
        initial_epochs = 100        
        history = self.model.fit(train_ds,
                epochs=initial_epochs,
                validation_data=val_ds,
                callbacks=[callbacks])
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([0.5,1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
        plt.savefig(name +  '.png')
        #### saving the plot

    def model_test(sel,checkpoint_path, x, inputs, model):
        model.load_weights(checkpoint_path)
        loss, acc = model.evaluate(test_ds, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


RESIZED = 100

name = ['mobilenetv2','xception', 'inceptionv3','resnet', 'vgg16','densenet']
preprocess = [tf.keras.applications.mobilenet_v2.preprocess_input,
        tf.keras.applications.xception.preprocess_input,
        tf.keras.applications.inception_v3.preprocess_input,
        tf.keras.applications.resnet_v2.preprocess_input,
        tf.keras.applications.vgg16.preprocess_input,
        tf.keras.applications.densenet.preprocess_input]
model = [tf.keras.applications.MobileNetV2(input_shape=(RESIZED, RESIZED) + (CHANNEL,),
    weights='imagenet',include_top=False),
    tf.keras.applications.Xception(input_shape=(RESIZED, RESIZED) + (CHANNEL,),
        weights="imagenet",include_top=False),
    tf.keras.applications.InceptionV3(input_shape=(RESIZED, RESIZED) + (CHANNEL,),
        weights="imagenet",include_top=False),
    tf.keras.applications.ResNet50V2(input_shape=(RESIZED, RESIZED) + (CHANNEL,),
        weights="imagenet", include_top=False),
    tf.keras.applications.VGG16(input_shape=(RESIZED, RESIZED) + (CHANNEL,),
            weights="imagenet", include_top=False),
    tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, 
        input_shape=(RESIZED, RESIZED) + (CHANNEL,))]

for i in range(0, len(name)):
    checkpoint_path = name[i] + '/' + name[i] + '_step1.ckpt'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_best_only=True)]
    test = NeuralNetwork_Performance(preprocess[i], model[i], callbacks, RESIZED)
    train_ds, val_ds, test_ds = test.import_data(15199,5066,5066)
    x, inputs = test.data_preprocessing()
    model1 = test.model_first_step_compile(x, inputs, train_ds)
    test.model_eval(train_ds, val_ds, callbacks, name[i] + "_step1")
    test.model_test(checkpoint_path, x, inputs, model1)
    checkpoint_path = name[i] + '/' + name[i] + '_step2.ckpt'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath= checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_best_only=True)]
    model1 = test.model_second_step_compile(x, inputs, train_ds)
    test.model_eval(train_ds, val_ds, callbacks, name[i])
    test.model_test(checkpoint_path, x, inputs, model1)

