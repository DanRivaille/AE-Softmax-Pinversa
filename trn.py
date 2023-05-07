# SNN's Training :

import numpy as np
import utility as ut


# Save weights and MSE  of the SNN
def save_w_dl(W_ae, W_sf, ann_MSE):
  np.savez('w_snn.npz', *W_ae)
  np.savez('w_snn.npz', W_sf)
  np.savetxt("costo.csv", ann_MSE)


def create_momentum(W, L):
  W_size = L + 1
  V = [None] * W_size

  for i in range(1, W_size):
    V[i] = np.zeros_like(W[i])

  return V


def get_minibatch(x, y, n, M):
  lower_bound = n * M
  upper_bound = (n + 1) * M

  x_batch = x[:, lower_bound: upper_bound]
  y_batch = y[:, lower_bound: upper_bound]

  return x_batch, y_batch


# miniBatch-SGDM's Training
def trn_minibatch(x, y, ann, param, V):
  N = len(x[0])
  M = param['M_batch']
  nBatch = N // M
  ann_mse = []
  min_mse = 10
  W = None

  for n in range(0, nBatch):
    xe, ye = get_minibatch(x, y, n, M)

    act = ut.forward(ann, param, xe)

    e = act - ye
    cost = ut.get_mse(act, ye)
    ann_mse.append(cost)

    if cost < min_mse:
      W = ann['W']
      min_mse = cost

    de_dw = ut.gradW(ann, param, e)
    ann['W'], V = ut.updWV_rmsprop(ann, param, de_dw, V)

  ann['W'] = W
  return ann_mse


# SNN's Training
def train(x, y, ann, param):
  V = create_momentum(ann['W'], ann['L'])
  mse = []

  for i in range(param['max_iter']):
    X, Y = ut.sort_data_random(x, y, x.shape[0])
    ann_mse = trn_minibatch(X, Y, ann, param, V)
    mse.append(np.mean(ann_mse))

    if (i % 100) == 0:
      print('\n Iterar-SGD: ', i, mse[i])

  print('\n Iterar-SGD: ', param['max_iter'], mse[-1])

  return ann['W'], mse


"""
# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, param):
  costo = []
  for i in range(numBatch):
    ...
    ...
  return (W, V, costo)





# AE's Training with miniBatch
def train_ae_batch(x, w1, v, w2, param):
  numBatch = np.int16(np.floor(x.shape[1] / param[0]))
  cost = []
  for i in range(numBatch):
    ....
  return (w1, v, cost)
    """


# Softmax's training via SGD with Momentum
def train_softmax(x, y):
  W = ut.iniW(y.shape[0], x.shape[0])
  """
  
  V = np.zeros(W.shape)
  ...
  for Iter in range(1, par1[0]):
    idx = np.random.permutation(x.shape[1])
    xe, ye = x[:, idx], y[:, idx]
    W, V, c = train_sft_batch(xe, ye, W, V, param)
    ...

  return (W, Costo)
  """
  return W


def init_ann(hidden_nodes, d, m):
  """
  Initialize an ANN with its variables saved into a map.
  :param hidden_nodes: List with the nodes quantity by layer.
  :param d: Size of the input
  :param m: Size of the output
  """
  ann = ut.create_ann(hidden_nodes)
  ann['W'] = ut.iniWs(ann['W'], ann['L'], d, m, hidden_nodes)

  return ann


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, param_ae, Ni):
  d = x.shape[0]
  ae = init_ann([Ni], d, d)
  """
  ....
  for Iter in range(1, param):
    xe = x[:, np.random.permutation(x.shape[1])]
    w1, v, c = train_ae_batch(xe, w1, v, w2, param)
    ....
  """

  return ae['W'][1]


# SAE's Training
def train_sae(X, param_ae):
  Wae = [None]
  xe = X
  for n_nodes in param_ae['ae_nodes']:
    W = train_ae(xe, param_ae, n_nodes)
    xe = W @ xe

    Wae.append(W)

  return Wae, xe

# Load data to train the SNN
def load_data_trn():
  FILE_X = 'dtrn.csv'
  FILE_Y = 'etrn.csv'
  X_train, y_train = ut.load_data(FILE_X, FILE_Y)
  return X_train, y_train


# Beginning ...
def main():
  param_ae = ut.load_cnf_ae()
  param_soft = ut.load_cnf_softmax()
  xe, ye = load_data_trn()
  W_ae, xe = train_sae(xe, param_ae)
  W_sf = train_softmax(xe, ye)
  #W_sf, Cost = train_softmax()
  np.savez('w_ae.npz', *W_ae)
  np.savez('w_sf.npz', W_sf)
  #save_w_dl(W_ae, W_sf, Cost)


if __name__ == '__main__':
  main()
