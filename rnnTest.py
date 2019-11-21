import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling1D, Convolution1D


from tensorflow.python.keras.optimizers import SGD
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np
from utils import *
import numpy


def parser(record):
    keys_to_features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([1480], tf.string),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = parsed['image_raw']
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)/255
    image = tf.reshape(image, [1,2960])


    label = tf.cast(parsed['label'], tf.int64)
    label = tf.one_hot(label, Label_size)

    return image, label

def dataset_input_fn():
    # filenames = ['I:\dataSet/result/*.tfrecord']
    path = 'I:\dataSet/result/zspilt/*.tfrecord'
    print(path)
    files = glob.glob(path)
    print(files)

    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=50000)
    #dataset = tf.data.Dataset.from_tensor_slices(files)
    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    #
    dataset = dataset.shuffle(buffer_size=10)

    #df = pd.DataFrame(dataset.list_files())
    #print(df)
    return dataset


def testdataset_input_fn():
    # filenames = ['I:\dataSet/result/*.tfrecord']
    path = 'I:\dataSet/result/Youtube.tfrecord'
    files = glob.glob(path)
    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=1000)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    #dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(10)

    return dataset

def Cnn(dataset,testdataset):
    model = Sequential()


    """x = dataset.make_one_shot_iterator()
    X=x.get_next()[0]
    X= tf.reshape(X, [-1,2960, 1])
    print("%s" ,X)
    Y = x.get_next()[1]
    Y =tf.reshape(Y, [-1,15])

    df = pd.DataFrame(x[0])
    print(df)"""

    ####################
    # 模型加入【輸入層】與【第一層卷積層】
    ####################

    model.add(Conv1D(8,2, activation='relu', input_shape=(1,2960),padding='same'))

    #model.add(MaxPooling1D(2))
    #model.add(Dropout(0.2))
    #model.add(Conv1D(12, 4, activation='relu',padding='same'))
    #model.add(MaxPooling1D(2))
    #model.add(Dropout(0.2))

    #model.add(Conv1D(32, 4, activation='relu',padding='same'))
    #model.add(MaxPooling1D(2))
    #model.add(Dropout(0.1))

    #model.add(Flatten())
    model.add(GlobalAveragePooling1D())

    model.add(Dense(15, activation='softmax'))

    ####################
    # 設定模型的訓練方式
    ####################

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy'  # 設定 Loss 損失函數 為 categorical_crossentropy
                  , optimizer= 'adam' # 設定 Optimizer 最佳化方法 為 adam
                  , metrics=['accuracy']  # 設定 Model 評估準確率方法 為 accuracy
    )


    history = model.fit(dataset, epochs= 7
              , steps_per_epoch= 1000 ,validation_data=testdataset, validation_steps=1,shuffle=True)

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

    model.save('I:\dataSet/my_model.h5')

    return model





if __name__ == '__main__':

    Label_size = 15

    dataset = dataset_input_fn()
    testdataset = testdataset_input_fn()
    dataset = dataset.batch(400).repeat()
    # 這是參數預測的結果

    testset = dataset.take(1000)
    trainset = dataset.skip(1000)

    #trainset = dataset.shard(2, 0)
    #testset = dataset.shard(2, 1)


    #traindata,testdata =train_test_split(dataset, test_size=0.3)

    model = Cnn(trainset,testset)

    """iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                e=sess.run(one_element)
                #print("%s\n" %e[0])
                #print("%s\n" %e[1])
                df = pd.DataFrame(e[1])
                print(df)
            except tf.errors.OutOfRangeError:
            print("end!")"""












