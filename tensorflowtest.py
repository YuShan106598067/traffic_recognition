import tensorflow as tf
from os import listdir
from os.path import join
import dpkt
import glob
import errno
import binascii
import csv


"""def class_text_to_int(row_label):
    if row_label == 'UDP':
        return 1
    elif row_label == 'TCP':
        return 2
    elif row_label == 'ICMP6':
        return 3
    elif row_label == 'ICMP':
        return 4
    else:
        return 0"""

def printSegment(f,idx,tfwriter):
    pcap = dpkt.pcapng.Reader(f)

    for (ts, buf) in pcap:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            segment = binascii.hexlify(ip.__bytes__())
            #segment = ip.data.__bytes__()
            #proto = ip.get_proto(ip.p).__name__

            info = [segment[i:i + 2] for i in range(0, len(segment), 2) if i < 2916]
            info.extend(b'00' for _ in range(1458 - len(info)))
            #info.extend((b'\x00\x00') for _ in range(1480 - len(info)))


            #writer.writerow([proto] + info)

            # print(info)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=info)),
                    }
                )
            )
            tfwriter.write(record=example.SerializeToString())
        except:
            pass


if __name__ == '__main__':

    #with open('C:/Users/janice/Desktop/output1.csv', 'w', newline='') as csvfile:

    #csvfile.close()
    #writer = csv.writer(csvfile)

    mypath = "I:\dataSet/training dataset"

    # 取得所有檔案與子目錄名稱
    dfiles = listdir(mypath)

    # 以迴圈處理
    for idx, f in enumerate(dfiles):
        # 產生檔案的絕對路徑
        print("檔案：", f.title())
        print(idx)
        #if f.title()=='Aim' :
        path = join(mypath, f)+'/*.pcapng'
        files = glob.glob(path)

        tfwriter = tf.python_io.TFRecordWriter('I:\dataSet/resultB/test/'+f.title()+'ng.tfrecord')
        for name in files:
            try:
                with open(name, 'rb') as f:
                    printSegment(f, idx, tfwriter)
                    f.close()
            except IOError as exc:
                if exc.errno != errno.EISDIR:

                    raise
    tfwriter.close()