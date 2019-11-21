
import tensorflow as tf
import pandas as pd
import numpy as np

import glob
import pandas as pd
from print_result import *
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import load_model
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sn

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

Label_size = 15
class_name = ["Aim","Email","Facebook","Gmail","Hangout","Icq","Netflix",
              "Scpdown","Sftpdown","Skype","Spotify","Tottwitter","Vimeo","Voipbuster","Youtube"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize  else 'd'
    thresh = cm.max() / .5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def extract_fn(record):
    keys_to_features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([1458], tf.string),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed['image_raw'], tf.uint8) / 255
    image = tf.reshape(image, [54, 54, 1])
    #print(image)
    label = tf.cast(parsed['label'], tf.int64)
    label = tf.one_hot(label, Label_size)
    # label = tf.reshape(label, (1, -1))
    return image, label

def plt_cm(df_cm):
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4, )
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="BuPu", fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print('aaaa')


path = 'I:\dataSet/resultA/test/testdataset/*.tfrecord'
files = glob.glob(path)
print(files)
dataset = tf.data.TFRecordDataset(files, num_parallel_reads=10000)

dataset = dataset.map(extract_fn)
#dataset = dataset.batch(10)
testset = dataset.take(15000)
testset = testset.batch(15000)

iterator = testset.make_one_shot_iterator()
next_element = iterator.get_next()


print("Using loaded model to predict...")
load_model = load_model('I:\dataSet/test_model_new1.h5')
predicted = load_model.predict(next_element[0], steps=1)
y_classes = predicted
df = pd.DataFrame(y_classes)
df['max_value'] = df.max(axis=1)
df['position'] = df.idxmax(axis=1)


y_pred = np.argmax(predicted, axis=1)


count = df[df['max_value'] > 0.5].groupby('position').size()

Y_pred = label_binarize(y_pred, classes=[i for i in range(15)])



with tf.Session() as sess:
    try:
        while True:
            data_record = sess.run(next_element)

            image = data_record[0]
            label = data_record[1]
            print(label)
            print(y_pred)
            y_true = np.argmax(label, axis=1)
            roc_predict(Y_pred, label)
            df_cm =confusion_matrix(y_true, y_pred)

            #df_cm= pd.crosstab(y_true, y_pred, rownames=['實際值'], colnames=['預測值'])
            # plot_confusion_matrix(df_cm, classes=class_name, normalize=True,title=' confusion matrix')

            df_cm = pd.DataFrame(df_cm,index=class_name,columns=class_name)
            plt_cm(df_cm)


            exit(0)
    except:
        pass
    exit()