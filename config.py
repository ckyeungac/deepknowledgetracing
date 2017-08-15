import os
import csv
import random
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc

import logging
logger = logging.getLogger(__name__)

# specify the gpu device
# import os
# from Tools.utils import _make_dir, load_options
# options = load_options('options.json')
# os.environ["CUDA_DEVICE_ORDER"] = "OCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[2]:

DATA_DIR = './data/'
#train_file = os.path.join(DATA_DIR, 'builder_train.csv')
#test_file = os.path.join(DATA_DIR, 'builder_test.csv')
train_file = os.path.join(DATA_DIR, '0910_b_train.csv')
test_file = os.path.join(DATA_DIR, '0910_b_test.csv')


# In[3]:

def read_data_from_csv(filename):
    rows = []
    max_num_problems_answered = 0
    num_problems = 0
    
    print("Reading {0}".format(filename))
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    print("{0} lines was read".format(len(rows)))
    
    # tuples stores the student answering sequence as 
    # ([num_problems_answered], [problem_ids], [is_corrects])
    tuples = []
    for i in range(0, len(rows), 3):
        # numbers of problem a student answered
        num_problems_answered = int(rows[i][0])
        
        # only keep student with at least 3 records.
        if num_problems_answered < 3:
            continue
        
        problem_ids = rows[i+1]
        is_corrects = rows[i+2]
        
        invalid_ids_loc = [i for i, pid in enumerate(problem_ids) if pid=='']        
        for invalid_loc in invalid_ids_loc:
            del problem_ids[invalid_loc]
            del is_corrects[invalid_loc]
        
        tup =(num_problems_answered, problem_ids, is_corrects)
        tuples.append(tup)
        
        if max_num_problems_answered < num_problems_answered:
            max_num_problems_answered = num_problems_answered
        
        pid = max(int(pid) for pid in problem_ids if pid!='')
        if num_problems < pid:
            num_problems = pid
    # add 1 to num_problems because 0 is in the pid
    num_problems+=1

    #shuffle the tuple
    random.shuffle(tuples)

    print ("max_num_problems_answered:", max_num_problems_answered)
    print ("num_problems:", num_problems)
    print("The number of students is {0}".format(len(tuples)))
    print("Finish reading data.")
    
    return tuples, max_num_problems_answered, num_problems


# In[4]:

def padding(student_tuple, target_length):
    num_problems_answered = student_tuple[0]
    question_seq = student_tuple[1]
    question_corr = student_tuple[2]
    
    pad_length = target_length - num_problems_answered
    question_seq += [-1]*pad_length
    question_corr += [0]*pad_length
    
    new_student_tuple = (num_problems_answered, question_seq, question_corr)
    return new_student_tuple


# In[5]:

students_train, max_num_problems_answered_train, num_problems_train = read_data_from_csv(train_file)

students_train = [padding(student_tuple, max_num_problems_answered_train) 
                  for student_tuple in students_train]

students_test, max_num_problems_answered_test, num_problems_test = read_data_from_csv(test_file)

students_test = [padding(student_tuple, max_num_problems_answered_train) 
                  for student_tuple in students_test]


# ## Student Model
# 
# ### Placeholder Explanation
# X is the one-hot encoded input sequence of a student.
# y is the one-hot encoded correct sequence of a student.
# 
# For example, the student i has a seq [1, 3, 1, 2, 2] with correct map [0, 1, 1, 0, 0]. The X_seq will be one hot encoded as:
# $$
# \left[
#     \begin{array}{ccccc}
#         0&1&0&0&0\\
#         0&0&0&1&0\\
#         0&1&0&0&0\\
#         0&0&1&0&0\\
#     \end{array}
# \right]
# $$
# 
# The X_corr map will be one hot encoded as:
# $$
# \left[
#     \begin{array}{ccccc}
#         0&0&0&0&0\\
#         0&0&0&1&0\\
#         0&1&0&0&0\\
#         0&0&0&0&0\\
#     \end{array}
# \right]
# $$
# 
# Our desire $X^i$ will be encoded as the following:
# $$
# \left[
#     \begin{array}{ccccc}
#         0&-1&0&0&0\\
#         0&0&0&1&0\\
#         0&1&0&0&0\\
#         0&0&-1&0&0\\
#     \end{array}
# \right]
# $$
# 
# 
# The last question '2' is not used in the $X^i$ because it is the last record that the student has and therefore used in $y$.
# So, $y$ would be seq [3, 1, 2, 2] with corr map [1, 1, 0, 0]
# $$
# \left[
#     \begin{array}{ccccc}
#         0&0&0&1&0\\
#         0&1&0&0&0\\
#         0&0&0&0&0\\
#         0&0&0&0&0\\
#     \end{array}
# \right]
# $$
# 

# In[6]:

def seq_corr_to_onehot(seq, corr, num_steps, num_problems):
    seq_oh = tf.one_hot(seq, depth=num_problems)
    seq_oh_flat = tf.reshape(seq_oh, [-1, num_problems])
    
    # element-wise multiplication between Matrix and Vector
    # the i-th column of Matrixelement-wisedly multiply the i-th element in the Vector
    corr_flat = tf.reshape(corr, [-1])
    corr_mat = tf.multiply(tf.transpose(seq_oh_flat), tf.cast(corr_flat, dtype=tf.float32))
    corr_mat = tf.transpose(corr_mat)
    corr_mat = tf.reshape(corr_mat, shape=[-1, num_steps, num_problems])
    
    corr_mat_value_two = corr_mat * 2
    
    X = corr_mat_value_two - seq_oh
    
    return seq_oh, corr_mat, X


# In[7]:

# network configuration
batch_size = 32
num_layers = 2
state_size = 200
num_steps = max_num_problems_answered_train-1
num_problems = num_problems_train
keep_prob = tf.placeholder(tf.float32)

inputs_seq = tf.placeholder(tf.int32, [None, num_steps])
inputs_corr = tf.placeholder(tf.int32, [None, num_steps])
X_seq, X_corr, X = seq_corr_to_onehot(inputs_seq, inputs_corr, num_steps, num_problems)

targets_seq = tf.placeholder(tf.int32, [None, num_steps])
targets_corr = tf.placeholder(tf.int32, [None, num_steps])
y_seq, y_corr, _ = seq_corr_to_onehot(targets_seq, targets_corr, num_steps, num_problems)

init_state = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])
state_per_layer_list  = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(
            state_per_layer_list[idx][0],
            state_per_layer_list[idx][1]
        ) for idx in range(num_layers)])


# In[8]:

X


# ### Network Configuration
# There are basically 2 elements needed to construct the LSTM network
# 1. The cell, and
# 2. The rnn structure.
# 
# The cell is defined via the tf.contrib.rnn library. It supports the multilayer RNN as well. 
# 
# The RNN is defined via the tf.nn.dynamic_rnn. It is parameterized by the cell defined, the input X, and a initial state.

# In[9]:

# build up the network
with tf.variable_scope('cell'):
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.LSTMCell(num_units=state_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True)

        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                            output_keep_prob=keep_prob)
        
        cells.append(cell)
    
    cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

with tf.variable_scope('rnn'):
    states_series, current_state = tf.nn.dynamic_rnn(cells, 
                                                    X,
                                                    initial_state=rnn_tuple_state,
                                                    time_major=False)

print("the states series is:\n", states_series)
print("\nthe current_state is:\n", current_state)


# In[ ]:

# this code block calculate the loss using tf.gather_nd
W_yh = tf.Variable(tf.random_normal([state_size, num_problems]), name="W_yh")
b_yh = tf.Variable(tf.constant(0.1, shape=[num_problems,]), name="b_yh")

states_series = tf.reshape(states_series, [-1, state_size])
logits_flat = tf.matmul(states_series, W_yh) + b_yh
y_seq_flat = tf.cast(tf.reshape(y_seq, [-1, num_problems]), dtype=tf.float32)
y_corr_flat = tf.cast(tf.reshape(y_corr, [-1, num_problems]), dtype=tf.float32)

# get the indices where they are not equal to 0
# the indices implies that a student has answered the question in the time step
# and thereby exclude those time step that the student hasn't answered.
target_indices = tf.where(tf.not_equal(y_seq_flat, 0))
target_logits = tf.gather_nd(logits_flat, target_indices)
target_labels = tf.gather_nd(y_corr_flat, target_indices)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits, 
                                               labels=target_labels)
total_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)


# In[ ]:

def optimize(sess):
    students = students_train
    
    # update the network configuration
    global num_steps
    num_steps = max_num_problems_answered_train - 1
    
    for epoch_idx in range(num_epochs):
        y_pred = []
        y_true = []
        num_students = len(students[:10])
        iteration = 0
        for batch_idx in range(0, num_students, batch_size):
            start_idx = batch_idx
            end_idx = min(num_students, batch_idx+batch_size)
            
            new_batch_size = end_idx - start_idx
            _current_state = np.zeros((num_layers, 2, new_batch_size, state_size))
            
            inputs_seq_batch = np.array([tup[1][:-1] for tup in students[start_idx:end_idx]], dtype=np.int32)
            inputs_corr_batch = np.array([tup[2][:-1] for tup in students[start_idx:end_idx]], dtype=np.int32)
            
            y_seq_batch = np.array([tup[1][1:] for tup in students[start_idx:end_idx]], dtype=np.int32)
            y_corr_batch = np.array([tup[2][1:] for tup in students[start_idx:end_idx]], dtype=np.int32)

            _optimizer, _current_state, = sess.run(
                    [optimizer, current_state],
                    feed_dict={
                    inputs_seq: inputs_seq_batch,
                    inputs_corr: inputs_corr_batch,
                    targets_seq: y_seq_batch,
                    targets_corr: y_corr_batch,
                    init_state: _current_state,
                    keep_prob: 0.5,
                })
            
            if iteration%10 == 0:
                _total_loss= sess.run(total_loss,
                    feed_dict={
                    inputs_seq: inputs_seq_batch,
                    inputs_corr: inputs_corr_batch,
                    targets_seq: y_seq_batch,
                    targets_corr: y_corr_batch,
                    init_state: _current_state,
                    keep_prob: 1,
                })
                print("Epoch {0:>4}, iteration {1:>4}, batch loss value: {2:.5}".format(epoch_idx, iteration, _total_loss))
            
            iteration+=1
        auc_train = evaluate(sess, is_train=True)
        auc_test = evaluate(sess, is_train=False)
        print("Epoch {0:>4}, Training AUC: {1:.5}, Testing AUC: {2:.5}".format(epoch_idx, auc_train, auc_test))
        

def evaluate(sess, is_train=False):
    global num_steps
    
    if is_train:
        students = students_train
        num_steps = max_num_problems_answered_train
    else:
        students = students_test
        num_steps = max_num_problems_answered_test

    
    y_pred = []
    y_true = []
    num_students = len(students[:10])
    for batch_idx in range(0, num_students, batch_size):
        start_idx = batch_idx
        end_idx = min(num_students, batch_idx+batch_size)

        new_batch_size = end_idx - start_idx
        _current_state = np.zeros((num_layers, 2, new_batch_size, state_size))

        inputs_seq_batch = np.array([tup[1][:-1] for tup in students[start_idx:end_idx]], dtype=np.int32)
        inputs_corr_batch = np.array([tup[2][:-1] for tup in students[start_idx:end_idx]], dtype=np.int32)

        y_seq_batch = np.array([tup[1][1:] for tup in students[start_idx:end_idx]], dtype=np.int32)
        y_corr_batch = np.array([tup[2][1:] for tup in students[start_idx:end_idx]], dtype=np.int32)

        _target_logits, _target_labels = sess.run(
                [target_logits, target_labels],
                feed_dict={
                inputs_seq: inputs_seq_batch,
                inputs_corr: inputs_corr_batch,
                targets_seq: y_seq_batch,
                targets_corr: y_corr_batch,
                init_state: _current_state,
                keep_prob: 1,
            })

        y_pred += [p for p in _target_logits]
        y_true += [t for t in _target_labels]

    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score


# In[ ]:

WITH_CONFIG = True
num_epochs = 25

start_time = time.time()
logger.info("Start the program...")
if WITH_CONFIG:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        optimize(sess)
else:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimize(sess)
           
end_time = time.time()

print("program run for: {0}s".format(end_time-start_time))


# In[ ]:



