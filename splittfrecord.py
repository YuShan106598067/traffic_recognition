import tensorflow as tf
import tensorflow as tf
from os import listdir
from os.path import join
import dpkt
import glob
import errno
import binascii

def split_tfrecord(tfrecord_path, split_size , filename,idx):
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:

                records = sess.run(batch)
                part_path = "I:\dataSet/resultB" \
                            "/test/zspilt/" + '{:03d}'.format(part_num) + filename
                with tf.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break

if __name__ == '__main__':


    mypath = "I:\dataSet/resultB/test"

    # 取得所有檔案與子目錄名稱
    dfiles = listdir(mypath)

    # 以迴圈處理
    for idx, f in enumerate(dfiles):
        # 產生檔案的絕對路徑
        fullpath = join(mypath, f)
        print("檔案：", fullpath)
        split_tfrecord(fullpath, 100, f,idx)