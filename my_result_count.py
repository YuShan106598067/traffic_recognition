import tensorflow as tf

import math
import glob
import pandas as pd
from print_result import *
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import load_model
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import seaborn as sn

class_name = ["Aim","Email","Facebook","Gmail","Hangout","Icq","Netflix",
              "Scpdown","Sftpdown","Skype","Spotify","Tottwitter","Vimeo","Voipbuster","Youtube"]

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def parser(record):
    keys_to_features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([1458], tf.string),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed['image_raw'], tf.uint8) / 255
    image = tf.reshape(image, [54, 54, 1])
    print(image)
    label = tf.cast(parsed['label'], tf.int64)
    label = tf.one_hot(label, Label_size)
    # label = tf.reshape(label, (1, -1))
    return image, label

def pre(load_model):


    predicted = load_model.predict(dataset, steps=1)
    #score = load_model.evaluate(dataset,steps=1)
    #print('Model test_acc:', score[1])



    y_classes = predicted
    df = pd.DataFrame(y_classes)
    df['max_value'] = df.max(axis=1)
    df['position'] = df.idxmax(axis=1)
    print(df)


    for i in range(int(batch_size/10)):
        print("round {:d}".format(i))
        batch = df[(i*10):(i*10+10)]
        print(batch)
        count = batch[batch['max_value'] > 0.7 ].groupby('position').max_value.sum()
        number = batch[batch['max_value'] > 0.5].groupby('position').max_value.size()
        print(number)
        print(number.sum())
        if(count.empty==False and number.max() > math.floor(10*0.3) ):
            flow = tf.one_hot([count.idxmax() for y in range(10)],Label_size)
            a = tf.Session().run(flow)
            print(a)
            predicted[(i*10):(i*10+10)] = a
        else :
            flow = tf.one_hot([15 for y in range(10)], Label_size)
            a = tf.Session().run(flow)
            print(a)
            predicted[(i*10):(i*10+10)] = a



    """for i in range(int(batch_size)):
        print("round {:d}".format(i))
        batch = df[i+1:i+2]


        if(batch[batch['max_value'] > 0.7 ].empty):
            flow = tf.one_hot([15 for y in range(1)], Label_size)
            a = tf.Session().run(flow)
            print(a)
            predicted[i] = a"""





    return predicted

def compare(dataset,predicted):

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    data_record = tf.Session().run(next_element)
    label = data_record[1]
    print(label)
    y_true = np.argmax(label, axis=1)
    y_pred = np.argmax(predicted, axis=1)
    df_cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(df_cm, index=class_name, columns=class_name)

    Y_pred = label_binarize(y_pred, classes=[i for i in range(Label_size)])

    roc_predict(Y_pred, label)


    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4, )
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="BuPu", fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print('aaaa')





if __name__ == '__main__':

    Label_size = 15
    batch_size = 1500
    #batch_size = 10
    print("Using loaded model to predict...")
    load_model = load_model('I:\dataSet/test_model_new1.h5')

    path = 'I:\dataSet/testing dataset/*.tfrecord'
    files = glob.glob(path)
    print(files)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)

    pre = pre(load_model)
    compare(dataset,pre)

    print("done")


