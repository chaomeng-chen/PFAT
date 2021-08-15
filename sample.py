import tensorflow as tf
import mnist_inference as mnist
import os
from DiffPrivate_FedLearning import run_differentially_private_federated_averaging
from MNIST_reader import Data
import argparse
import sys


def sample(N, b,e,m, sigma, eps, save_dir, log_dir, MODEL_NUM,THV, C, method, isTest):
    # Specs for the model that we would like to train in differentially private federated fashion:

    hidden1 = 600
    hidden2 = 100

    # Specs for the differentially private federated fashion learning process.

    # A data object that already satisfies client structure and has the following attributes:
    # DATA.data_set : A list of labeld training examples.
    # DATA.client_set : A
    DATA = Data(save_dir, N)

    with tf.Graph().as_default():

        # Building the model that we would like to train in differentially private federated fashion.
        # We will need the tensorflow training operation for that model, its loss and an evaluation method:

        train_op, eval_correct, loss, data_placeholder, labels_placeholder = mnist.mnist_fully_connected_model(b, hidden1, hidden2)
        #train_op, eval_correct, loss, data_placeholder, labels_placeholder = mnist.mnist_cnn_model(b)

        Accuracy_accountant, Delta_accountant, model = \
            run_differentially_private_federated_averaging(loss, train_op, eval_correct, DATA, data_placeholder,
                                                           labels_placeholder, b=b, e=e,m=m, sigma=sigma, eps=eps,
                                                           save_dir=save_dir, log_dir=log_dir, MODEL_NUM=MODEL_NUM,
                                                           THV=THV, C=C, method=method, isTest=isTest)

def main(_):
   if FLAGS.isTest==1:
        sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e, m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps,
                   save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir, MODEL_NUM=FLAGS.MODEL_NUM, THV=FLAGS.THV, C=FLAGS.C,
                   method=0, isTest=True)
   else:
       if FLAGS.paramTest==1:
           THV = [0.02, 0.05, 0.1, 0.5, 1, 2, 5, 10]
           for i in range(len(THV)):
               sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e, m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps,
                      save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir, MODEL_NUM=10, THV=THV, C=FLAGS.C,
                      method=0, isTest=False)
               sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e, m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps,
                      save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir, MODEL_NUM=K[i], THV=FLAGS.THV, C=FLAGS.C,
                      method=1, isTest=False)
               # sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e, m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps,
               #        save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir, MODEL_NUM=K[i], THV=FLAGS.THV, C=FLAGS.C,
               #        method=2, isTest=False)
       else:
           sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e, m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps,
               save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir, MODEL_NUM=FLAGS.MODEL_NUM, THV=FLAGS.THV, C=FLAGS.C,
               method=0, isTest=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.getcwd(),
        help='directory to store progress'
    )
    parser.add_argument(
        '--N',
        type=int,
        default=100,
        help='Total Number of clients participating'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0,
        help='The gm variance parameter; will not affect if Priv_agent is set to True'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=8,
        help='Epsilon'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=0,
        help='Number of clients participating in a round'
    )
    parser.add_argument(
        '--b',
        type=float,
        default=25,
        help='Batches per client'
    )
    parser.add_argument(
        '--e',
        type=int,
        default=4,
        help='Epochs per client'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--MODEL_NUM',
        type=int,
        default=10,
        help='Number of models.'
    )
    parser.add_argument(
        '--THV',
        type=float,
        default=1.0,
        help='Number of THV.'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=0,
        help='Number of C.'
    )
    parser.add_argument(
        '--method',
        type=int,
        default=0,
        help='0：本算法，1：无噪声，2：FedAvg'
    )
    parser.add_argument(
        '--paramTest',
        type=int,
        default=0,
        help='0：正常输入参数，1：THV=[0.5,5]间隔0.5，2：C=[0.01,0.1]间隔0.01'
    )
    parser.add_argument(
        '--isTest',
        type=int,
        default=0,
        help='0：训练模式，1：测试模式'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

