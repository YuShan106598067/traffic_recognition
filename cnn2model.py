
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Convolution2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,Dropout,GlobalAveragePooling2D,Activation
from tensorflow.python.keras.optimizers import SGD
import glob
import pandas as pd
import numpy as np
from utils import *
from tensorflow.python.keras.optimizers import SGD

def parser(record):
    keys_to_features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([1458], tf.string),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)/255

    image = tf.reshape(image, [54, 54, 1])

    label = tf.cast(parsed['label'], tf.int64)
    label = tf.one_hot(label, Label_size)
    # label = tf.reshape(label, (1, -1))
    return image, label

def dataset_input_fn():
    # filenames = ['I:\dataSet/result/*.tfrecord']
    path = 'I:\dataSet/resultA/test/zspilt/*.tfrecord'
    files = glob.glob(path)
    print(files)
    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=800000)
    #dataset = tf.data.Dataset.from_tensor_slices(files)
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.

    # Use `Dataset.map()` to build a pair of a feature dictionar
    # y and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    #
    dataset = dataset.shuffle(buffer_size=5000)

    #df = pd.DataFrame(dataset.list_files())
    #print(df)
    return dataset

def Cnn(dataset,testdataset):

    model = Sequential()

    model.add(Conv2D(filters=8,
                     kernel_size=(3, 3),
                     strides=(1,1),
                     padding='same',
                     input_shape=(54, 54, 1),
                     activation='relu'))
    # Create Max-Pool 1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Create CN layer 2
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu'))

    # Create Max-Pool 2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     padding='same'))

    model.add(Flatten())

    model.add(Dense(Label_size, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    ####################
    # 設定模型的訓練方式
    ####################
    model.compile(loss='categorical_crossentropy'  # 設定 Loss 損失函數 為 categorical_crossentropy
                  , optimizer= 'adam' # 設定 Optimizer 最佳化方法 為 adam
                  , metrics=['accuracy']  # 設定 Model 評估準確率方法 為 accuracy
    )


    history = model.fit(dataset, epochs= 30

              , steps_per_epoch= 1000 ,validation_data=testdataset, validation_steps=1,shuffle=True)

    model.save('I:\dataSet/test_model_unknow_hahaha.h5')

    score = model.evaluate(testdataset, steps=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    predicted = model.predict(testdataset, steps=1)
    print(predicted)
    y_classes = predicted
    df = pd.DataFrame(y_classes)
    print(df)

    show_train_history(history, 'acc', 'val_acc')
    show_train_history(history, 'loss', 'val_loss')



    return model






if __name__ == '__main__':

    Label_size = 15

    dataset = dataset_input_fn()
    #testdataset = testdataset_input_fn()
    dataset = dataset.batch(400).repeat()
    # 這是參數預測的結果

    testset = dataset
    trainset = dataset

    #trainset = dataset.shard(2, 0)
    #testset = dataset.shard(2, 1)


    #traindata,testdata =train_test_split(dataset, test_size=0.3)

    model = Cnn(trainset,testset)





