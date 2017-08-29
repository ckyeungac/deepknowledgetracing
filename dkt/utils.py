from dkt import model as DKTModel
import os, sys, time
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import math
from dkt.load_data import OriginalInputProcessor

SPLIT_MSG = "***********"

def get_model(Model, num_problem, model_name=None, **network_config):
    model = Model(
        num_problems=num_problem,
        **network_config
    )
    return model


class BasicDKT():
    def __init__(self, sess, data, network_config, **kwargs):
        self.sess = sess
        self.data_train = data.train
        self.data_test = data.test
        self.num_problems = data.num_problems
        self.network_config = network_config
        self.model = DKTModel.BasicModel(num_problems=data.num_problems, **network_config)
        self.keep_prob = kwargs.get('keep_prob', 0.5)
        self.num_epochs = kwargs.get('num_epochs', 500)
        self.num_runs = kwargs.get('num_runs', 5)
        self.save_dir = kwargs.get('save_dir', './checkpoint/latest/')

    def train(self):
        data = self.data_train
        model = self.model
        keep_prob = self.keep_prob
        sess = self.sess

        loss = 0
        y_pred = []
        y_true = []
        iteration = 1
        for batch_idx in range(data.num_batches):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch()
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: keep_prob,
            }
            _, _target_preds, _target_labels, _loss = sess.run(
                [model.train_op, model.target_preds, model.target_labels, model.loss],
                feed_dict=feed_dict
            )
            y_pred += [p for p in _target_preds]
            y_true += [t for t in _target_labels]
            loss = (iteration - 1) / iteration * loss + _loss / iteration
            iteration += 1
        try:
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            auc_score = auc(fpr, tpr)
        except ValueError:
            print("Value Error is encountered during finding the auc_score. Assign the AUC to 0 now.")
            auc_score = 0
        return auc_score, loss

    def evaluate(self):
        data = self.data_test
        model = self.model
        sess = self.sess

        y_pred = []
        y_true = []
        iteration = 1
        loss = 0
        for batch_idx in range(data.num_batches):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch()
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: 1,
            }
            _target_preds, _target_labels, _loss = sess.run(
                [model.target_preds, model.target_labels, model.loss],
                feed_dict=feed_dict
            )
            y_pred += [p for p in _target_preds]
            y_true += [t for t in _target_labels]
            loss = (iteration - 1) / iteration * loss + _loss / iteration
            iteration += 1
        try:
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            auc_score = auc(fpr, tpr)
        except ValueError:
            print("Value Error is encountered during finding the auc_score. Assign the AUC to 0 now.")
            auc_score = 0

        return auc_score, loss

    def run_optimization(self):
        num_epochs = self.num_epochs
        num_runs = self.num_runs
        save_dir = self.save_dir
        sess = self.sess

        total_auc = 0
        for run_idx in range(num_runs):
            sess.run(tf.global_variables_initializer())
            best_test_auc = 0
            best_epoch_idx = 0
            for epoch_idx in range(num_epochs):
                epoch_start_time = time.time()
                auc_train, loss_train = self.train()
                print(
                    'Epoch {0:>4}, Train AUC: {1:.5}, Train Loss: {2:.5}'.format(epoch_idx + 1, auc_train, loss_train))

                auc_test, loss_test = self.evaluate()
                test_msg = "Epoch {0:>4}, Test AUC: {1:.5}, Test Loss: {2:.5}".format(epoch_idx + 1, auc_test,
                                                                                      loss_test)
                if auc_test > best_test_auc:
                    test_msg += "*"
                    best_epoch_idx = epoch_idx
                    best_test_auc = auc_test
                    test_msg += ". Saving the model"
                    self.save_model()

                print(test_msg)
                epoch_end_time = time.time()
                print("time used for this epoch: {0}s".format(epoch_end_time - epoch_start_time))
                print(SPLIT_MSG)

                # quit the training if there is no improve in AUC for 10 epochs.
                if epoch_idx - best_epoch_idx >= 10:
                    print("No improvement shown in 10 epochs. Quit Training.")
                    break
                sys.stdout.flush()
                # shuffle the training dataset
                self.data_train.shuffle()
            print("The best testing result occured at: {0}-th epoch, with testing AUC: {1:.5}".format(best_epoch_idx,
                                                                                                      best_test_auc))
            print(SPLIT_MSG * 3)
            total_auc += best_test_auc
        avg_auc = total_auc / num_runs
        print("average AUC for {0} runs: {1:.5}".format(num_runs, avg_auc))

    def save_model(self):
        save_dir = self.save_dir
        sess = self.sess
        # Define the tf saver
        saver = tf.train.Saver()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'model')
        saver.save(sess=sess, save_path=save_path)

    def load_model(self):
        save_dir = self.save_dir
        sess = self.sess
        saver = tf.train.Saver()
        save_path = os.path.join(save_dir, 'model')
        if os.path.exists(save_path):
            saver.restore(sess=sess, save_path=save_path)
        else:
            print("No model found at {}".format(save_path))

    def get_hidden_layer_output(self, problem_seqs, correct_seqs, layer):
        model = self.model
        sess = self.sess
        num_layer = len(model.hidden_layer_structure)
        assert layer < num_layer, "There are only {0} layers. indexed from 0.".format(num_layer)

        input_processor = OriginalInputProcessor()
        X, y_seq, y_corr = input_processor.process_problems_and_corrects(problem_seqs=problem_seqs,
                                                                         correct_seqs=correct_seqs,
                                                                         num_problems=self.num_problems)

        feed_dict = {
            model.X: X,
            model.y_seq: y_seq,
            model.y_corr: y_corr,
            model.keep_prob: 1,
        }

        hidden_layers_outputs = sess.run(
            model.hidden_layers_outputs,
            feed_dict=feed_dict
        )

        result = hidden_layers_outputs[layer]
        return result

    def get_output_layer(self, problem_seqs, correct_seqs):
        model = self.model
        sess = self.sess

        input_processor = OriginalInputProcessor()
        X, y_seq, y_corr = input_processor.process_problems_and_corrects(problem_seqs=problem_seqs,
                                                                         correct_seqs=correct_seqs,
                                                                         num_problems=self.num_problems)

        feed_dict = {
            model.X: X,
            model.y_seq: y_seq,
            model.y_corr: y_corr,
            model.keep_prob: 1,
        }

        pred_seqs = sess.run(
            model.preds_flat,
            feed_dict=feed_dict
        )

        result = pred_seqs
        return result


class GaussianInputNoiseDKT(BasicDKT):
    def __init__(self, sess, data, network_config, **kwargs):
        super().__init__(sess, data, network_config, **kwargs)
        self.gaussian_std = 1.0/math.sqrt(data.num_problems)
        self.model = get_model(
            Model=DKTModel.GaussianNoiseInputModel,
            num_problem = data.num_problems,
            **network_config
        )

    def train(self):
        data = self.data_train
        model = self.model
        keep_prob = self.keep_prob
        gaussian_std = self.gaussian_std
        sess = self.sess

        loss = 0
        y_pred = []
        y_true = []
        iteration = 1
        for batch_idx in range(data.num_batches):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch()
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: keep_prob,
                model.gaussian_std: gaussian_std
            }
            _, _target_preds, _target_labels, _loss = sess.run(
                [model.train_op, model.target_preds, model.target_labels, model.loss],
                feed_dict=feed_dict
            )
            y_pred += [p for p in _target_preds]
            y_true += [t for t in _target_labels]
            loss = (iteration - 1) / iteration * loss + _loss / iteration
            iteration += 1
        try:
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            auc_score = auc(fpr, tpr)
        except ValueError:
            print("Value Error is encountered during finding the auc_score. Assign the AUC to 0 now.")
            auc_score = 0
        return auc_score, loss

    def evaluate(self):
        data = self.data_test
        model = self.model
        sess = self.sess

        y_pred = []
        y_true = []
        iteration = 1
        loss = 0
        for batch_idx in range(data.num_batches):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch()
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: 1,
                model.gaussian_std: 0

            }
            _target_preds, _target_labels, _loss = sess.run(
                [model.target_preds, model.target_labels, model.loss],
                feed_dict=feed_dict
            )
            y_pred += [p for p in _target_preds]
            y_true += [t for t in _target_labels]
            loss = (iteration - 1) / iteration * loss + _loss / iteration
            iteration += 1
        try:
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            auc_score = auc(fpr, tpr)
        except ValueError:
            print("Value Error is encountered during finding the auc_score. Assign the AUC to 0 now.")
            auc_score = 0

        return auc_score, loss


class ProblemEmbeddingDKT(BasicDKT):
    def __init__(self, sess, data, network_config, **kwargs):
        super().__init__(sess, data, network_config, **kwargs)
        self.embedding_size = kwargs.get('embedding_size', 200)
        self.model = DKTModel.ProblemEmbeddingModel(num_problems=data.num_problems,
                                                    embedding_size=self.embedding_size,
                                                    **network_config)