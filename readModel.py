import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow import keras

from os import listdir
from os.path import isfile, isdir, join


if __name__ == '__main__':

    mypath = "I:\dataSet/training dataset"

    # 取得所有檔案與子目錄名稱
    dfiles = listdir(mypath)

    # 以迴圈處理
    for f in dfiles:
        # 產生檔案的絕對路徑
        #fullpath = join(mypath, f)

        print(f.title())





