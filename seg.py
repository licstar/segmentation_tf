# -*- coding: UTF-8 –*-
import tensorflow as tf
import time
import math
import sys
import numpy as np
from datetime import datetime

PAD_ID = 0
UNK_ID = 1

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('batch_size', 4, 'batch size')

#TODO 读取embedding，只记录词表中的词
def read_embedding(file_name, vocab):
  with open(file_name) as embedding_f:
    for line in embedding_f:
      words = line.strip().split(" ") #strip().lower()

#创建词表
#TODO 未来版本参考Google的加上最小出现次数、词表大小等约束
def create_vocab(data_path):
  vocab = {"PAD": PAD_ID, "UNK": UNK_ID}
  with open(data_path) as data_f:
    for line in data_f:
      for node in line.strip().split(" "):
        word = node.split("/")[0]
        if not word in vocab:
          if create_vocab:
            vocab[word] = len(vocab)
  return vocab

class DataSet(object):

  def __init__(self, data_path, window_size, vocab, class_size):
    self._data, self._label = self.read_file(data_path, window_size, vocab)
    self._label = self.dense_to_one_hot(self._label, class_size)
    self._data = np.array(self._data)
    self._label = np.array(self._label)
    self._index_in_epoch = 0
    self._num_examples = len(self._data)
    self._epochs_completed = 0

  #输入
  #file_name文件名，window_size窗口大小，vocab词表
  #输入文件格式：在/0 这/1 个/2 激/1 动/3 人/3 心/2 的/0 时/1 刻/2 ，/0 我/0 很/0 高/1
  #返回
  #data  [0,0,在,这,个],[0,在,这,个,激],[在,这,个,激,动],[这,个,激,动,人],...
  #data会对应到词表编号
  #label [0,1,2,1,3,3,2,0,1,2...]
  def read_file(self, data_path, window_size, vocab):
    data = []
    label = []
    hw = (window_size-1)/2 #half window
    with open(data_path) as data_f:
      for line in data_f:
        data_line = [0, 0]
        for node in line.strip().split(" "):
          x = node.split("/")
          data_line.append(vocab.get(x[0], UNK_ID))
          label.append(int(x[1]) % 4) #数字、字母、混合可以/4得到
        data_line.extend([0, 0])
        for i in range(hw, len(data_line)-hw):
          data.append([data_line[i+j] for j in range(-hw, hw+1)])
    return data, label

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._label = self._label[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end], self._label[start:end]

  def dense_to_one_hot(self, labels_dense, class_size):
    ret = []
    for i in range(0, len(labels_dense)):
      line = [0] * class_size
      line[labels_dense[i]] = 1
      ret.append(line)
    return ret

  def get_data(self):
    return self._data

  def get_label(self):
    return self._label

  def get_epoch(self):
    return self._epochs_completed

def main(_):
  #read_embedding("embedding.txt")

  window_size = 5 #需要奇数
  vec_size = 50
  hidden_size = 100
  class_size = 4
  batch_size = int(FLAGS.batch_size)
  input_size = vec_size * window_size

  #读取文件
  vocab = create_vocab("seg_train.txt")

  train = DataSet("seg_train.txt", window_size, vocab, class_size)
  valid = DataSet("seg_valid.txt", window_size, vocab, class_size)
  test = DataSet("seg_test.txt", window_size, vocab, class_size)

  data_size = len(train.get_data())

  #所有变量
  embeddings = tf.Variable(tf.random_uniform([len(vocab), vec_size], 
    -0.5, 0.5))
  W1 = tf.Variable(tf.random_uniform([input_size, hidden_size], 
    -0.5/math.sqrt(input_size), 0.5/math.sqrt(input_size)))
  W2 = tf.Variable(tf.random_uniform([hidden_size, class_size], 
    -0.5/math.sqrt(hidden_size), 0.5/math.sqrt(hidden_size)))
  b1 = tf.Variable(tf.zeros([hidden_size]))
  b2 = tf.Variable(tf.zeros([class_size]))
  
  #输入输出
  ids = tf.placeholder("int32", shape=[None, window_size])
  y_ = tf.placeholder("float", shape=[None, class_size])

  #网络结构
  ids_embedding = tf.nn.embedding_lookup(embeddings, ids)
  x = tf.reshape(ids_embedding, [-1, vec_size * window_size])
  h = tf.tanh(tf.matmul(x, W1) + b1)
  y = tf.nn.softmax(tf.matmul(h, W2) + b2)
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  #其它用于记录的指标
  cross_entropy_mean = -tf.reduce_mean(y_*tf.log(y))
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


  #调试信息
  tf.histogram_summary('x', x)
  tf.histogram_summary('h', h)
  tf.histogram_summary('y', y)
  tf.histogram_summary('W1', W1)
  tf.histogram_summary('W2', W2)
  tf.histogram_summary('b1', b1)
  tf.histogram_summary('b2', b2)
  tf.scalar_summary('cross_entropy', cross_entropy_mean)
  tf.scalar_summary('accuracy', accuracy)


  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(("/tmp/seg3/%d" % batch_size), graph_def=sess.graph_def)
    
    duration_total = 0
    batch_count = 0

    #for step in xrange(10000000):
    step = 0
    while True:

        start_time = time.time()

        batch_data, batch_label = train.next_batch(batch_size)
        data_now = {ids: batch_data, y_: batch_label}
        sess.run([train_step], feed_dict=data_now)

        duration_total += time.time() - start_time
        batch_count += 1

        #记录log，一个epoch记录10次
        if step % (data_size/(batch_size*10)) == 0:
          train_acc, loss = sess.run([accuracy, cross_entropy_mean], feed_dict={ids: train.get_data(), y_: train.get_label()})
          valid_acc = sess.run(accuracy, feed_dict={ids: valid.get_data(), y_: valid.get_label()})
          test_acc = sess.run(accuracy, feed_dict={ids: test.get_data(), y_: test.get_label()})
          
          examples_per_sec = batch_size * batch_count / float(duration_total)
          sec_per_batch = float(duration_total) / batch_count
          duration_total = 0
          batch_count = 0

          format_str = ('%s: step %d, epoch %d, loss = %.2f acc = %.2f %.2f %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
          print (format_str % (datetime.now(), step, train.get_epoch(),
                  loss, train_acc*100, valid_acc*100, test_acc*100,
                  examples_per_sec, sec_per_batch))
          sys.stdout.flush()

          summary_str = sess.run(summary_op, feed_dict={ids: train.get_data(), y_: train.get_label()})
          summary_writer.add_summary(summary_str, 
            step / (data_size/(batch_size*10)))

        step += 1

if __name__ == '__main__':
  tf.app.run()
