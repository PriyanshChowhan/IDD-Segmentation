import tensorflow as tf
import cv2
from imutils import paths
from tqdm import tqdm
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import datetime
from glob import glob
import IPython.display as display
from IPython.display import clear_output
import math
import time
from tensorflow.keras.layers import *
import warnings
warnings.filterwarnings('ignore')
AUTOTUNE = tf.data.experimental.AUTOTUNE


SEED = 42

dataset_path = 'drive/My Drive/idd20k_lite/'
img_train = dataset_path + 'leftImg8bit/train/'
seg_train = dataset_path + 'gtFine/train/'

img_val = dataset_path + 'leftImg8bit/val/'
seg_val = dataset_path + 'gtFine/val/'

# Reading a sample image and plotting it
img = cv2.imread(img_train+'87/357164_image.jpg',1)
plt.imshow(img)

def visualizeSegmentationImages(imagepath):
    img_seg = cv2.imread(imagepath,0)
    for i in range(len(img_seg)):
        for j in range(len(img_seg[0])):
            if img_seg[i][j] != 0 or img_seg[i][j] != 255:
                img_seg[i][j] *= 40
    return img_seg

img_seg = visualizeSegmentationImages(seg_train+'87/357164_label.png')
plt.imshow(img_seg)

(HEIGHT,WIDTH) = (128,256)
N_CHANNELS = 3
N_CLASSES = 8

TRAINSET_SIZE = len(glob(img_train+'*/*_image.jpg'))
print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

VALSET_SIZE = len(glob(img_val+'*/*_image.jpg'))
print(f"The Validation Dataset contains {VALSET_SIZE} images.")

def parse_image(img_path):
    """
    Load an image and its annotation (mask) and returning a dictionary.
    """

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    mask_path = tf.strings.regex_replace(img_path, "leftImg8bit", "gtFine")
    mask_path = tf.strings.regex_replace(mask_path, "_image.jpg", "_label.png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask==255, np.dtype('uint8').type(7), mask)
    return {'image': image, 'segmentation_mask': mask}

train_dataset = tf.data.Dataset.list_files(img_train+'*/*_image.jpg', seed=SEED)
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(img_val+'*/*_image.jpg', seed=SEED)
val_dataset = val_dataset.map(parse_image)

def normalize(input_image, input_mask):
    """
    Rescale the pixel values of the images between 0 and 1 compared to [0,255] originally.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

def load_image_train(datapoint):
    """
    Normalize and resize a train image and its annotation.
    Apply random transformations to an input dictionary containing a train image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (HEIGHT,WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (HEIGHT,WIDTH))
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_image_test(datapoint):
    """
    Normalize and resize a test image and its annotation.
    Since this is for the test set, we don't need to apply any data augmentation technique.
    """
    input_image = tf.image.resize(datapoint['image'], (HEIGHT,WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (HEIGHT,WIDTH))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

BATCH_SIZE = 32
BUFFER_SIZE = 1500
dataset = {"train": train_dataset, "val": val_dataset}

dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
print(dataset['train'])

# Preparing the Validation Dataset
dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
print(dataset['val'])

def display_sample(display_list):
    """
    Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(15,15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Getting a sample image for visualizing
for image, mask in dataset['train'].take(2):
    sample_image, sample_mask = image, mask

display_sample([sample_image[0], sample_mask[0]])

from tensorflow.keras.layers import Conv2D as Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

"""Model"""

num_epochs = 10

input_shape = (128, 256, 3)
num_classes = 8
drop_rate = 0.5

BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
STEPS_PER_EPOCH

# Build UNet
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
inputs = layers.Input(input_shape)
conv1 = layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = layers.BatchNormalization()(conv1)
conv1 = layers.Activation("relu")(conv1)
conv1 = layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
conv1 = layers.BatchNormalization()(conv1)
conv1 = layers.Activation("relu")(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = layers.BatchNormalization()(conv2)
conv2 = layers.Activation("relu")(conv2)
conv2 = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
conv2 = layers.BatchNormalization()(conv2)
conv2 = layers.Activation("relu")(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = layers.BatchNormalization()(conv3)
conv3 = layers.Activation("relu")(conv3)
conv3 = layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
conv3 = layers.BatchNormalization()(conv3)
conv3 = layers.Activation("relu")(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = layers.BatchNormalization()(conv4)
conv4 = layers.Activation("relu")(conv4)
conv4 = layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
conv4 = layers.BatchNormalization()(conv4)
conv4 = layers.Activation("relu")(conv4)
drop4 = layers.Dropout(drop_rate)(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = layers.Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = layers.BatchNormalization()(conv5)
conv5 = layers.Activation("relu")(conv5)
conv5 = layers.Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
conv5 = layers.BatchNormalization()(conv5)
conv5 = layers.Activation("relu")(conv5)
drop5 = layers.Dropout(drop_rate)(conv5)

up6 = layers.Conv2DTranspose(1024, 2, padding='same', strides=(2, 2))(drop5)
merge6 = layers.concatenate([drop4, up6], axis=3)
conv6 = layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = layers.BatchNormalization()(conv6)
conv6 = layers.Activation("relu")(conv6)
conv6 = layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
conv6 = layers.BatchNormalization()(conv6)
conv6 = layers.Activation("relu")(conv6)

up7 = layers.Conv2DTranspose(512, 2, padding='same', strides=(2, 2))(conv6)
merge7 = layers.concatenate([conv3, up7], axis=3)
conv7 = layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = layers.BatchNormalization()(conv7)
conv7 = layers.Activation("relu")(conv7)
conv7 = layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
conv7 = layers.BatchNormalization()(conv7)
conv7 = layers.Activation("relu")(conv7)

up8 = layers.Conv2DTranspose(256, 2, padding='same', strides=(2, 2))(conv7)
merge8 = layers.concatenate([conv2, up8], axis=3)
conv8 = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = layers.BatchNormalization()(conv8)
conv8 = layers.Activation("relu")(conv8)
conv8 = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
conv8 = layers.BatchNormalization()(conv8)
conv8 = layers.Activation("relu")(conv8)

up9 = layers.Conv2DTranspose(128, 2, padding='same', strides=(2, 2))(conv8)
merge9 = layers.concatenate([conv1, up9], axis=3)
conv9 = layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = layers.BatchNormalization()(conv9)
conv9 = layers.Activation("relu")(conv9)
conv9 = layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = layers.BatchNormalization()(conv9)
conv9 = layers.Activation("relu")(conv9)

conv10 = layers.Conv2D(num_classes, 1, activation = 'softmax')(conv9)

model = tf.keras.Model(inputs=inputs, outputs=conv10)

# Defining a loss object and an optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, 'tf_ckpts/', max_to_keep=3)

# Define the metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

@tf.function
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y_train, predictions)

def train_and_checkpoint(model, manager, dataset, epoch):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    for (x_train, y_train) in dataset['train'].take(math.ceil(1403/32)):
        train_step(model, optimizer, x_train, y_train)
    ckpt.step.assign_add(1)
    save_path = manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

@tf.function
def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    test_loss(loss)
    test_accuracy(y_test, predictions)
    return predictions

train_log_dir = 'logs/gradient_tape/train'
test_log_dir = 'logs/gradient_tape/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

highest_accuracy = 0

for epoch in range(50):

    print("Epoch ",epoch+1)

    # Getting the current time before starting the training
    # This will help to keep track of how much time an epoch took
    start = time.time()

    train_and_checkpoint(model, manager, dataset, epoch+1)

    # Saving the train loss and train accuracy metric for TensorBoard visualization
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=ckpt.step.numpy())
        tf.summary.scalar('accuracy', train_accuracy.result(), step=ckpt.step.numpy())

    # Validation phase
    for (x_test, y_test) in dataset['val'].take(math.ceil(204/32)):
        pred = test_step(model, x_test, y_test)

    # Saving the validation loss and validation accuracy metric for Tensorboard visualization
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=ckpt.step.numpy())
        tf.summary.scalar('accuracy', test_accuracy.result(), step=ckpt.step.numpy())

    # Calculating the time it took for the entire epoch to run
    print("Time taken ",time.time()-start)

    # Printing the metrics for the epoch
    template = 'Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}'
    print (template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100))

    # If accuracy has increased in this epoch, updating the highest accuracy and saving the model
    if(test_accuracy.result().numpy()*100>highest_accuracy):
        print("Validation accuracy increased from {:.3f} to {:.3f}. Saving model weights.".format(highest_accuracy,test_accuracy.result().numpy()*100))
        highest_accuracy = test_accuracy.result().numpy()*100
        model.save_weights('unet_weights-epoch-{}.weights.h5'.format(epoch+1))

    print('_'*80)

    # Reset metrics after every epoch
    train_loss.reset_state()
    test_loss.reset_state()
    train_accuracy.reset_state()
    test_accuracy.reset_state()

os.listdir()

model.load_weights('unet_weights-epoch-34.weights.h5')

def predict(model,image_path):
    """
    This function will take the model which is going to be used to predict the image and the image path of
    the input image as inputs and predict the mask
    It returns the true mask and predicted mask
    """
    datapoint = parse_image(image_path)
    input_image,image_mask = load_image_test(datapoint)
    img = tf.expand_dims(input_image, 0)
    prediction = model(img)
    prediction = tf.argmax(prediction, axis=-1)
    prediction = tf.squeeze(prediction, axis = 0)
    pred_mask = tf.expand_dims(prediction, axis=-1)
    return image_mask, pred_mask

true_mask, pred_mask = predict(model,'drive/My Drive/idd20k_lite/leftImg8bit/val/21/240284_image.jpg')

def IoU(y_i,y_pred):
    IoUs = []
    n_classes = 8
    for c in range(n_classes):
        TP = np.sum((y_i == c)&(y_pred==c))
        FP = np.sum((y_i != c)&(y_pred==c))
        FN = np.sum((y_i == c)&(y_pred!= c))
        IoU = TP/float(TP + FP + FN)
        if(math.isnan(IoU)):
            IoUs.append(0)
            continue
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    return mIoU

IoU(true_mask, pred_mask)

highest_accuracy

model.load_weights('unet_weights-epoch-34.weights.h5')

model.save_weights('drive/My Drive/idd20k_lite/unet_weights-epoch-34.weights.h5')

img_val = dataset_path + 'leftImg8bit/val/'
val_paths = glob(img_val+'*/*_image.jpg')

mIoU = []
for path in val_paths:
    true_mask, pred_mask = predict(model,path)
    mIoU.append(IoU(true_mask, pred_mask))

print("Validation mIoU = ",sum(mIoU)/len(mIoU))

img_test = dataset_path + 'leftImg8bit/test/'
test_paths = glob(img_test+'*/*_image.jpg')

test_images_final = []
test_images_name = []

for imagePath in test_paths:
    image = cv2.imread(imagePath)
    make = imagePath.split("/")[-1]
    filename = make[:make.rfind("_")]
    make_new = imagePath.split("/")[-2]
    directory="output/"+ make_new
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_name = "output/"+ make_new+"/"+filename+"_label.png"
    test_images_final.append(image)
    test_images_name.append(image_name)

test_images_final = np.array(test_images_final)

test_images_final.shape

for i,image in enumerate(test_images_final):
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.resize(image, (HEIGHT,WIDTH))
    image = tf.expand_dims(image, 0)
    prediction = model(image)
    prediction = tf.argmax(prediction, axis=-1)
    prediction = tf.squeeze(prediction, axis = 0)
    prediction = tf.expand_dims(prediction, axis=-1)
    prediction = np.array(prediction)
    prediction[prediction == 7] = 255
    cv2.imwrite(test_images_name[i],prediction)
