# SNN's Training :

import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse(W, ann_MSE):
  np.savez('w_snn.npz', *W)
  np.savetxt("costo.csv", np.array(ann_MSE))


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

#miniBatch-SGDM's Training 
def trn_minibatch(x, y, ann, param):
  N = len(x[0])
  M = param['M_batch']
  V = create_momentum(ann['W'], ann['L'])
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
    ann['W'], V = ut.updWV_sgdm(ann, param, de_dw, V)

  ann['W'] = W
  return ann_mse

#SNN's Training 
def train(x, y, param):
  ann = init_ann(param, x)
  mse = []

  for i in range(param['max_iter']):
    X, Y = ut.sort_data_random(x,y, x.shape[0])
    ann_mse = trn_minibatch(X, Y, ann, param)
    mse.append(np.mean(ann_mse))

    if (i % 100) == 0:
      print('\n Iterar-SGD: ', i, mse[i])

  print('\n Iterar-SGD: ', param['max_iter'], mse[-1])

  return ann['W'], mse


def init_ann(param, x):
  ann = ut.create_ann(param['hidden_nodes'], x)
  d = x.shape[0]
  ann['W'] = ut.iniWs(ann['W'], ann['L'], d, param['n_classes'], param['hidden_nodes'])

  return ann


# Load data to train the SNN
def load_data_trn(param):
  FILE_X = 'dtrn.csv'
  FILE_Y = 'etrn.csv'
  X_train, y_train = ut.load_data(FILE_X, FILE_Y, param['n_classes'])
  return X_train, y_train


# Beginning ...
def main():
  param = ut.load_cnf()            
  xe, ye = load_data_trn(param)
  W, Cost = train(xe, ye, param)
  save_w_mse(W, Cost)


if __name__ == '__main__':   
  main()

