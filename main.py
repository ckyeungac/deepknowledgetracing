from dkt.load_data import ASSISTment2009
from dkt.utils import BasicDKT, GaussianInputNoiseDKT
import os
import tensorflow as tf
import time

"""
Assignable variables:
num_runs: int
num_epochs: int
keep_prob: float
is_early_stopping: boolean
early_stopping: int
batch_size: int
hidden_layer_structure: tuple
data_dir: str
train_file_name: str
test_file_name: str
ckpt_save_dir: str
"""


DATA_DIR = './data/'
train_file = 'skill_id_train.csv'
test_file = 'skill_id_test.csv'
train_path = os.path.join(DATA_DIR, train_file)
test_path = os.path.join(DATA_DIR, test_file)

network_config = {
    'batch_size': 32,
    'hidden_layer_structure': (200,),
    'rnn_cell': tf.contrib.rnn.LSTMCell,
    'learning_rate': 0.01,
    'keep_prob': 0.5,
}


def main():
    data = ASSISTment2009(train_path, test_path, batch_size=32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model_name = 'latest'
    save_dir = './checkpoints/' + model_name + '/'
    # initialize model
    dkt = GaussianInputNoiseDKT(sess=sess,
                   data=data,
                   network_config=network_config,
                   num_epochs=1000,
                   num_runs=5,
                   save_dir=save_dir)

    # run optimization of the created model
    dkt.model.build_graph()
    dkt.run_optimization()

    # close the session
    sess.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("program run for: {0}s".format(end_time - start_time))