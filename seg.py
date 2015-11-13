# -*- coding: UTF-8 –*-
import tensorflow as tf
import time
from datetime import datetime

PAD_ID = 0
UNK_ID = 1

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


#输入
#file_name文件名，window_size窗口大小，vocab词表
#输入文件格式：在/0 这/1 个/2 激/1 动/3 人/3 心/2 的/0 时/1 刻/2 ，/0 我/0 很/0 高/1
#返回
#data  [0,0,在,这,个],[0,在,这,个,激],[在,这,个,激,动],[这,个,激,动,人],...
#data会对应到词表编号
#label [0,1,2,1,3,3,2,0,1,2...]
def read_file(data_path, window_size, vocab):
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

'''
def loss(logits, labels, batch_size, class_size):
  sparse_labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  onehot_labels = tf.sparse_to_dense(concated, [batch_size, class_size], 1.0, 0.0)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels)
  return tf.reduce_mean(cross_entropy, name='cross_entropy')
'''
#def step():

#TODO 先搞个能跑但是不对的版本
batch_pos = 0
def get_batch(data, label, batch_size):
  batch_data = []
  batch_label = []
  global batch_pos
  for i in range(0, batch_size):
    batch_data.append(data[batch_pos])
    batch_label.append(label[batch_pos])
    batch_pos = batch_pos + 1
    if batch_pos == len(data):
      batch_pos = 0
  return batch_data, batch_label

def dense_to_one_hot(labels_dense, class_size):
  ret = []
  for i in range(0, len(labels_dense)):
    line = [0] * class_size
    #print len(line), labels_dense[i]
    line[labels_dense[i]] = 1
    ret.append(line)
  return ret

if __name__ == "__main__":
  #read_embedding("embedding.txt")

  window_size = 5 #需要奇数
  vec_size = 50
  hidden_size = 50
  class_size = 4
  batch_size = 32
  input_size = vec_size * window_size

  #读取文件
  vocab = create_vocab("seg_train.txt")
  data_train, label_train = read_file("seg_train.txt", window_size, vocab)
  data_valid, label_valid = read_file("seg_valid.txt", window_size, vocab)
  data_test, label_test = read_file("seg_test.txt", window_size, vocab)
  
  label_train = dense_to_one_hot(label_train, class_size) #TODO 先用这个顶着，看看怎么优化
  label_test = dense_to_one_hot(label_test, class_size)
  label_valid = dense_to_one_hot(label_valid, class_size)

  print len(label_train)

  #所有变量
  embeddings = tf.Variable(tf.zeros([len(vocab), vec_size]))
  W1 = tf.Variable(tf.zeros([input_size, hidden_size]))
  W2 = tf.Variable(tf.zeros([hidden_size, class_size]))
  b1 = tf.Variable(tf.zeros([hidden_size]))
  b2 = tf.Variable(tf.zeros([class_size]))
  
  #输入输出
  ids = tf.placeholder("int32", shape=[None, window_size])
  y_ = tf.placeholder("float", shape=[None, class_size])

  #生成batch
  #data_train = tf.convert_to_tensor(data_train)
  #data_train = tf.train.slice_input_producer(data_train)
  #label_train = tf.train.slice_input_producer(label_train)
  #data_batch, label_batch = tf.train.shuffle_batch(
  #   [data_train, label_train], batch_size,
  #    capacity=50000, min_after_dequeue=10000)

  #网络结构
  ids_embedding = tf.nn.embedding_lookup(embeddings, ids)
  x = tf.reshape(ids_embedding, [-1, vec_size * window_size])
  h = tf.tanh(tf.matmul(x, W1) + b1)
  y = tf.nn.softmax(tf.matmul(h, W2) + b2)
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))

  #tf.histogram_summary('cross_entropy', cross_entropy)
  #cross_entropy = loss(logits, y_, batch_size, class_size)


  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  

  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  #tf.train.start_queue_runners(sess=sess)

  summary_writer = tf.train.SummaryWriter("/tmp/seg", graph_def=sess.graph_def)

  for step in xrange(10000000):
      start_time = time.time()
      batch_data, batch_label = get_batch(data_train, label_train, batch_size)
      #print tf.convert_to_tensor(batch_data).get_shape().as_list()
      train_step.run(session=sess, feed_dict={ids: batch_data, y_: batch_label})

      if step % 10000 == 0: #一次扫描结束
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        loss = cross_entropy.eval(session=sess, feed_dict={ids: data_train, y_: label_train})
        train_acc = accuracy.eval(session=sess, feed_dict={ids: data_train, y_: label_train})
        valid_acc = accuracy.eval(session=sess, feed_dict={ids: data_valid, y_: label_valid})
        test_acc = accuracy.eval(session=sess, feed_dict={ids: data_test, y_: label_test})
        #_, loss_value = sess.run([train_op, loss])

        duration = time.time() - start_time
        examples_per_sec = batch_size / float(duration)
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f %.2f %.2f %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
        print (format_str % (datetime.now(), step, loss, train_acc, valid_acc, test_acc,
                               examples_per_sec, sec_per_batch))

        #summary_str = sess.run(tf.merge_all_summaries())
        #summary_writer.add_summary(summary_str, step)