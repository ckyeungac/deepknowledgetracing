# deepknowledgetracing
Tensorflow implementation for the [Deep Knowledge Tracing](http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)

To use the original DKT model. You could use the following command:
```
python main.py
```

----------
This project enable the extensability of using the DKT. It provides:
- Specify the hidden layer structure by using '-hl', e.g. '-hl 200 50' implies using 200 units in the first layer and 50 units in the second layer.
- Specify whether to use gaussian noise on the input data by flating '-gn'.
- Specify the RNN Cell that to used in the network by '-cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}'
- Far more, please look into the below section of program usage.


# Program usage
----------
```
usage: main.py [-h] [--num_runs NUM_RUNS] [--num_epochs NUM_EPOCHS]
               [--batch_size BATCH_SIZE]
               [-hl [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]]]
               [-gn] [-cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}]
               [-lr LEARNING_RATE] [--keep_prob KEEP_PROB]
               [--ckpt_dir CKPT_DIR] [--model_name MODEL_NAME]
               [--data_dir DATA_DIR] [--train_file TRAIN_FILE]
               [--test_file TEST_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --num_runs NUM_RUNS   Number of runs to repeat the experiment.
  --num_epochs NUM_EPOCHS
                        Maximum number of epochs to train the network.
  --batch_size BATCH_SIZE
                        The mini-batch size used when training the network.
  -hl [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]], --hidden_layer_structure [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_
STRUCTURE ...]]
                        The hidden layer structure in the RNN. If there is 2
                        hidden layers with first layer of 200 and second layer
                        of 50. Type in '-hl 200 50'
  -gn, --use_gaussian_noise
                        Flag this to add gaussian noise to the input.
  -cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}, --rnn_cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}
                        Specify the rnn cell used in the graph.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate when training the model.
  --keep_prob KEEP_PROB
                        Keep probability when training the network.
  --ckpt_dir CKPT_DIR   The base directory that the model parameter going to
                        store.
  --model_name MODEL_NAME
                        The directory that the model parameter going to store.
  --data_dir DATA_DIR   the data directory, default as './data/
  --train_file TRAIN_FILE
                        train data file, default as 'skill_id_train.csv'.
  --test_file TEST_FILE
                        train data file, default as 'skill_id_test.csv'.

```
