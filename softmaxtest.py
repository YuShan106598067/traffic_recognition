import tensorflow as tf


def read_and_decode(filename, batch_size):
    # 建立文件名隊列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    # 數據讀取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 數據解析
    img_features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([1480], tf.string), })

    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [1, 2960])

    label = tf.cast(img_features['label'], tf.int64)
    # label = tf.reshape(label, (-1, -1))

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size=batch_size,
                                 capacity=10000 + 3 * batch_size,
                                 min_after_dequeue=1000)

    return image_batch, label_batch

# 自己做好的 TF 檔在哪裡，自己知道
filename = 'C:/Users/janice/Desktop/output.tfrecord'

# batch 可以自由設定
batch_size = 256

# 0-9共10個類別，請根據自己的資料修改
Label_size = 5

# 調用剛才的函數
image_batch, label_batch = read_and_decode(filename, batch_size)

# 轉換陣列的形狀
image_batch_train = tf.reshape(image_batch, [-1, 1*2960])

# 把 Label 轉換成獨熱編碼
label_batch_train = tf.one_hot(label_batch, Label_size)
# label_batch_train = tf.reshape(label_batch_train, (1, -1))

# W 和 b 就是我們要訓練的對象
W = tf.Variable(tf.zeros([1*2960, Label_size]))
b = tf.Variable(tf.zeros(Label_size))

# 我們的影像資料，會透過 x 變數來輸入
x = tf.placeholder(tf.float32, [None, 1*2960])

# 這是參數預測的結果
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 這是每張影像的正確標籤
y_ = tf.placeholder(tf.float32,[None,5])

# 計算最小交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

# 使用梯度下降法來找最佳解
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

####################################################

# # 計算預測正確率
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 計算 y 向量的最大值
y_pred = tf.argmax(y, 1)

# 建立 tf.train.Saver 物件
saver = tf.train.Saver()

# 將輸入與輸出值加入集合
tf.add_to_collection('input', x)
tf.add_to_collection('output', y_pred)

####################################################

with tf.Session() as sess:
    # 初始化是必要的動作
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 建立執行緒協調器
    coord = tf.train.Coordinator()

    # 啟動文件隊列，開始讀取文件
    threads = tf.train.start_queue_runners(coord=coord)

    # 迭代 10000 次，看看訓練的成果
    for count in range(100):
        # 這邊開始讀取資料
        image_data, label_data = sess.run([image_batch_train, label_batch_train])
        print(label_data)
        # 送資料進去訓練
        sess.run(train_step, feed_dict={x: image_data, y_: label_data})

        # # 這裡是結果展示區，每 10 次迭代後，把最新的正確率顯示出來
        if count % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
            print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy * 100))
        exit(0)
    # 結束後記得把文件名隊列關掉
    coord.request_stop()
    coord.join(threads)


 ####################################################
    # 這裡也是新增的內容 #

    # 存檔路徑 #
    """save_path = 'C:/Users/janice/Desktop/test_model'

    # 把整張計算圖存檔
    spath = saver.save(sess, save_path)
    print("Model saved in file: %s" % spath)"""
    ####################################################


