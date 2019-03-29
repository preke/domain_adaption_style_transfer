import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
import time
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


DATA_HOME = './data/'
THRESHOLD = 0.0001
STEP_SIZE = 0.00001


def safe_log(x, nan_substitute=-1e+4):
    l = np.log(x)
    l[np.logical_or(np.isnan(l), np.isinf(l))] = nan_substitute
    return l

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function_safe(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -sum(y * safe_log(h) + (1 - y) * safe_log(1 - h))


def get_gradient(theta, X, y):
    errors = sigmoid(X.dot(theta)) - y
    return errors.dot(X)

def accuracy(theta, X, y):
    correct = np.sum(np.equal(y, (sigmoid(X.dot(theta))) > 0.5))
    return correct / y.size

def get_binary_data():
    mnist            = fetch_mldata('MNIST original', data_home='./data/')
    df_mnist         = pd.DataFrame(pd.concat([pd.DataFrame(mnist['data']), pd.Series(mnist['target'])], axis=1))
    df_mnist.columns = [str(i) for i in list(range(1, 785))] + ['label']
    df_mnist_binary  = df_mnist[df_mnist['label'] < 2]
    df_mnist_binary  = shuffle(df_mnist_binary)
    df_train         = df_mnist_binary[:int(0.9*len(df_mnist_binary))]
    df_test          = df_mnist_binary[int(0.9*len(df_mnist_binary)):]
    train_X, test_X  = normalize_features(df_train.iloc[:, :-1], df_test.iloc[:, :-1])
    train_X          = pd.concat([ pd.DataFrame([1]* len(train_X)), pd.DataFrame(np.array(train_X))], axis=1, ignore_index=True)
    test_X           = pd.concat([ pd.DataFrame([1]* len(test_X)) , pd.DataFrame(np.array(test_X))], axis=1, ignore_index=True)
    train_X          = np.array(train_X)
    test_X           = np.array(test_X)
    train_y          = np.array(df_train['label'])
    test_y           = np.array(df_test['label'])
    return train_X, train_y, test_X, test_y


def train(theta, train_X, train_y):
    step   = 0
    losses = []
    loss   = get_loss(theta, train_X, train_y)
    losses.append(loss)
    while True:
        gradient = get_gradient(theta, train_X, train_y)
        theta    = theta - gradient*STEP_SIZE
        loss     = get_loss(theta, train_X, train_y)
        if step%1000 == 0:
            logger.info('Loss in step %d is: %s'%(step, loss)) 
        losses.append(loss)
        step += 1
        if abs(losses[step] - losses[step-1]) < THRESHOLD:
            break

    return theta, losses


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = get_binary_data()
    m, n          = train_X.shape[0], train_X.shape[1]
    theta         = np.random.rand(n) * 0.001
    t_start       = time.time()
    theta, losses = train(theta, train_X, train_y)
    t_end         = time.time()
    logger.info('Training cost %f seconds.' %(t_end - t_start))
    logger.info('Training accuracy: {acc}'.format(acc=accuracy(theta, train_X, train_y)))
    logger.info('Test accuracy: {acc}'.format(acc=accuracy(theta, test_X, test_y)))




