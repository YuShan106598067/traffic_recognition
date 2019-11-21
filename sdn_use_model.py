
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling1D, Convolution1D
from tensorflow.python.keras.layers import Conv1D, Dense, Flatten,Dropout,GlobalAveragePooling1D,Activation

from os import listdir
from os.path import join
import dpkt
import glob
import errno
import binascii
from tensorflow.python.keras.models import load_model
import pandas as pd
import numpy as np

def printSegment(f,idx,model):
    pcap = dpkt.pcap.Reader(f)

    for (ts, buf) in pcap:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            segment = binascii.hexlify(ip.data.__bytes__())
            #segment = ip.data.__bytes__()
            #proto = ip.get_proto(ip.p).__name__

            info = [segment[i:i + 2] for i in range(0, len(segment), 2) if i < 2916]
            info.extend(b'00' for _ in range(1458 - len(info)))
            #print(info)

            image = np.asarray(info)
            image = np.fromstring(image, dtype=np.uint8)/255

            #image = np.reshape(image, [1,1,2960])
            image = np.reshape(image, [1,54, 54,1])
            #

            print(image)

            predicted = model.predict(image)

            print("\nPredicted softmax vector is: ")
            print(predicted)

            y_classes = predicted
            df = pd.DataFrame(y_classes)
            df['max_value'] = df.max(axis=1)
            df['position'] = df.idxmax(axis=1)
            print(df)

        except:
            pass
        exit()

if __name__ == '__main__':

    mypath = "I:\dataSet/training dataset"

    # 取得所有檔案與子目錄名稱
    dfiles = listdir(mypath)
    load_model = load_model('I:\dataSet/test_model_unknow5.h5')
    load_model.summary()
    # 以迴圈處理｢
    for idx, f in enumerate(dfiles):
        # 產生檔案的絕對路徑
        print("檔案：", f.title())
        if f.title() == 'Email':
            path = join(mypath, f)+'/email2a.pcap'
            files = glob.glob(path)

            for name in files:
                try:
                    with open(name, 'rb') as f:
                        printSegment(f, idx,load_model)
                        f.close()
                except IOError as exc:
                    if exc.errno != errno.EISDIR:

                        raise
            #exit()


