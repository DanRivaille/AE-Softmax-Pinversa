# SNN's Training :

import numpy as np
import utility as ut


# Save weights and MSE  of the SNN
def save_w_dl(W_ae, W_sf, ann_MSE):
  np.savez('w_ae.npz', *W_ae)
  np.savez('w_sf.npz', W_sf)
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


def cross_entropy_cost(a, y):
  M = y.shape[1]
  cost = - (np.sum(y * np.log(a)) / M)
  return cost


# Calculate Softmax
def softmax(z):
  exp_z = np.exp(z - np.max(z))
  return exp_z / exp_z.sum(axis=0, keepdims=True)


# Training miniBatch for softmax
def train_sft_batch(x, y, W, V, S, param):
  N = x.shape[1]
  M = param['M_batch']
  nBatch = N // M
  mse = []

  for n in range(nBatch):
    xe, ye = get_minibatch(x, y, n, M)

    z = W @ xe
    act = softmax(z)
    e = ye - act
    costo = cross_entropy_cost(act, ye)
    mse.append(costo)

    gW = - ((e @ xe.T) / M)
    W, V, S = ut.applyAdam(param['mu'], V, S, gW, W, n)

  return W, V, S, mse


# Softmax's training via SGD with Momentum
def train_softmax(x, y, param):
  W = ut.iniW(y.shape[0], x.shape[0])
  V = np.zeros_like(W)
  S = np.zeros_like(W)
  Costo = []

  for i in range(param['max_iter']):
    idx = np.random.permutation(x.shape[1])
    xe, ye = x[:, idx], y[:, idx]
    W, V, S, cost = train_sft_batch(xe, ye, W, V, S, param)

    Costo.append(np.mean(cost))

    if (i % 50) == 0:
      print(i, Costo[-1])

  return W, Costo


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


# miniBatch-SGDM's Training
def trn_minibatch(x, y, ann, param, V, S):
  N = x.shape[1]
  M = param['M_batch']
  nBatch = N // M
  mse = []
  for n in range(nBatch):
    xe, ye = get_minibatch(x, y, n, M)

    act = ut.forward(ann, param, xe)
    e = act - ye
    costo = ut.get_mse(act, ye)
    mse.append(costo)
    de_dw = ut.gradW(ann, param, e)
    ann['W'], V, S = ut.updWV_rmsprop(ann, param, de_dw, V, S, n)

  #ann['W'][2] = ut.updPinv(ann, x, param)

  return ann['W'], V, S, mse


# AE's Training by use miniBatch RMSprop+Pinv
def train_ae(x, param_ae, Ni):
  d = x.shape[0]
  ae = init_ann([Ni], d, d)
  V = create_momentum(ae['W'], ae['L'])
  S = create_momentum(ae['W'], ae['L'])

  for i in range(param_ae['max_iter']):
    xe = x[:, np.random.permutation(x.shape[1])]
    ae['W'], V, S, mse = trn_minibatch(xe, xe, ae, param_ae, V, S)

    if (i % 10) == 0:
      print(i, np.mean(mse))

  return ae['W'][1]


# SAE's Training
def train_sae(X, param_ae):
  Wae = [None]
  xe = X
  for n_nodes in param_ae['ae_nodes']:
    W = train_ae(xe, param_ae, n_nodes)
    Wae.append(W)

    xe = ut.act_function(param_ae['g_fun'], W @ xe)

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
  W_sf, Cost = train_softmax(xe, ye, param_soft)
  save_w_dl(W_ae, W_sf, Cost)


if __name__ == '__main__':
  main()
