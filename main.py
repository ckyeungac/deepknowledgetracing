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
batch_size: int
hidden_layer_structure: tuple
data_dir: str
train_file_name: str
test_file_name: str
ckpt_save_dir: str
"""
import argparse
parser = argparse.ArgumentParser()
# training configuration
parser.add_argument("--num_runs", type=int, default=5,
                    help="Number of runs to repeat the experiment.")
parser.add_argument("--num_epochs", type=int, default=500,
                    help="Maximum number of epochs to train the network.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="The mini-batch size used when training the network.")
# network configuration
parser.add_argument("-hl", "--hidden_layer_structure", default=[200, ], nargs='*',
                    help="The hidden layer structure in the RNN. If there is 2 hidden layers with first layer "
                         "of 200 and second layer of 50. Type in '-hl 200 50'")
parser.add_argument("-gn", "--use_gaussian_noise", default=False, action='store_true',
                    help="Flag this to add gaussian noise to the input.")
parser.add_argument("-cell", "--rnn_cell", default='LSTM', choices=['LSTM', 'GRU', 'BasicRNN', 'LayerNormBasicLSTM'],
                    help='Specify the rnn cell used in the graph.')
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2,
                    help="The learning rate when training the model.")
parser.add_argument("--keep_prob", type=float, default=0.5,
                    help="Keep probability when training the network.")
# data file configuration
parser.add_argument("--ckpt_dir", type=str, default='./checkpoints/',
                    help="The base directory that the model parameter going to store.")
parser.add_argument("--model_name", type=str, default='latest',
                    help="The directory that the model parameter going to store.")
parser.add_argument('--data_dir', type=str, default='./data/')
parser.add_argument('--train_file', type=str, default='skill_id_train.csv')
parser.add_argument('--test_file', type=str, default='skill_id_test.csv')
args = parser.parse_args()

rnn_cells = {
    "LSTM": tf.contrib.rnn.LSTMCell,
    "GRU": tf.contrib.rnn.GRUCell,
    "BasicRNN": tf.contrib.rnn.BasicRNNCell,
    "LayerNormBasicLSTM": tf.contrib.rnn.LayerNormBasicLSTMCell,
}

train_path = os.path.join(args.data_dir, args.train_file)
test_path = os.path.join(args.data_dir, args.test_file)

network_config = {}
network_config['batch_size'] = args.batch_size
network_config['hidden_layer_structure'] = args.hidden_layer_structure
network_config['learning_rate'] = args.learning_rate
network_config['keep_prob'] = args.keep_prob
network_config['rnn_cell'] = rnn_cells[args.rnn_cell]

ckpt_base_dir = args.ckpt_dir
model_name = args.model_name
ckpt_save_dir = os.path.join(ckpt_base_dir, model_name)

num_epochs = args.num_epochs
num_runs = args.num_runs

def get_DKT():
    if args.use_gaussian_noise:
        return GaussianInputNoiseDKT
    else:
        return BasicDKT

def main():
    print("Network Configuration:\n", network_config)
    data = ASSISTment2009(train_path, test_path, batch_size=32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # initialize model
    DKT = get_DKT()
    dkt = DKT(sess=sess,
               data=data,
               network_config=network_config,
               num_epochs=num_epochs,
               num_runs=num_runs,
               save_dir=ckpt_save_dir)

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
