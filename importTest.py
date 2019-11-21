
import tensorflow as tf
import pandas as pd
import numpy as np


# dataset = tf.data.TextLineDataset("output.tfrecord")
#
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(10):
#         print(sess.run(next_element))


def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([1458], tf.string)
    }
    sample = tf.parse_single_example(data_record, features)

    return sample

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['I:\dataSet/000Skype.tfrecord'])
# filename_queue = tf.train.string_input_producer('C:/Users/janice/Desktop/Voip.tfrecord')

dataset = dataset.map(extract_fn)
dataset = dataset.batch(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# label_batch_train = tf.one_hot(label_batch, Label_size)
#tfwriter = tf.python_io.TFRecordWriter('I:\dataSet/resultB/test/new.tfrecord')
i=0



with tf.Session() as sess:
    try:
         while True:

            data_record = sess.run(next_element)

            image = data_record['image_raw']
            label = data_record['label']

            la =15
            print(i)
            """if (i== 0):
                for x in range(10):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=image[x])),
                            }
                        )
                    )
                    tfwriter.write(record=example.SerializeToString())
            if (i== 1):
                for x in range(10):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=image[x])),
                            }
                        )
                    )
                    tfwriter.write(record=example.SerializeToString())
            if (i== 2):
                for x in range(10):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=image[x])),
                            }
                        )
                    )
                    tfwriter.write(record=example.SerializeToString())
            if (i== 3):
                for x in range(10):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=image[x])),
                            }
                        )
                    )
                    tfwriter.write(record=example.SerializeToString())
            if (i== 8):
                for x in range(10):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=image[x])),
                            }
                        )
                    )
                    tfwriter.write(record=example.SerializeToString())
            if (i== 9):
                for x in range(10):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
                                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=image[x])),
                            }
                        )
                    )
                    tfwriter.write(record=example.SerializeToString())"""

            i=i+1
            print(data_record)

    except:
        pass

    #tfwriter.close()
    exit(0)
    exit()