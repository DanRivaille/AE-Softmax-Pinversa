# My Utility : auxiliars functions

import numpy as np

# Constantes
_alpha_elu = 0.1
_alpha_selu = 1.6732
_lambda = 1.0507
SIGMOID_N_FUN = 5
LINEAR_N_FUN = 0


def load_data(file_x, file_y):
  X = np.loadtxt(file_x, delimiter=',', dtype=float)
  y = np.loadtxt(file_y, delimiter=',', dtype=int)
  return X, y.T


def load_cnf_ae():
  FILE_CNF = 'cnf_sae.csv'
  param = dict()
  cnf_list = np.loadtxt(FILE_CNF, dtype=float)
  param['p_inv_param'] = int(cnf_list[0])
  param['g_fun'] = int(cnf_list[1])
  param['max_iter'] = int(cnf_list[2])
  param['M_batch'] = int(cnf_list[3])
  param['mu'] = cnf_list[4]
  
  ae_nodes = []

  for n_nodes in cnf_list[5:]:
    ae_nodes.append(int(n_nodes))

  param['ae_nodes'] = ae_nodes
  return param


def load_cnf_softmax():
  FILE_CNF = 'cnf_softmax.csv'
  param = dict()
  cnf_list = np.loadtxt(FILE_CNF, dtype=float)
  param['max_iter'] = int(cnf_list[0])
  param['mu'] = cnf_list[1]
  param['M_batch'] = int(cnf_list[2])
  return param


# Initialize weights for SNN-SGDM
def iniWs(W, L, d, m, n_nodes):
  W[1] = iniW(n_nodes[0], d)
  W[L] = iniW(m, n_nodes[-1])

  for i in range(L - 2):
    W[i + 2] = iniW(n_nodes[i + 1], n_nodes[i])

  return W


# Initialize weights for one-layer    
def iniW(next, prev):
  r = np.sqrt(6 / (next + prev))
  w = np.random.rand(next, prev)
  w = w * 2 * r - r
  return w


# Create a dictionary with the ann info
def create_ann(hidden_nodes):
  n_layers = len(hidden_nodes) + 1
  W = [None] * (n_layers + 1)  # Weights matrixes
  a = [None] * (n_layers + 1)  # activation matrixes
  z = [None] * (n_layers + 1)  # transfer matrixes

  ann = {'W': W, 'a': a, 'z': z, 'L': n_layers, 'hidden_nodes': hidden_nodes}
  return ann


# Feed-forward of SNN
def forward(ann, param, x):
  L = ann['L']
  w = ann['W']
  a = ann['a']
  z = ann['z']

  a[0] = x

  for i in range(1, L + 1):
    if (i == L):
      n_fun = SIGMOID_N_FUN
    else:
      n_fun = param['g_fun']

    z[i] = np.clip(w[i] @ a[i - 1], -50, 50)
    a[i] = act_function(n_fun, z[i])

  return a[L]


# Activation function
def act_function(num_function, x):
  if 0 == num_function:  # Linear
    return x
  if 1 == num_function:  # Relu
    return np.maximum(0, x)
  if 2 == num_function:  # L-Relu
    return np.maximum(0.01 * x, x)
  if 3 == num_function:  # ELU
    return np.maximum(_alpha_elu * (np.exp(x) - 1), x)
  if 4 == num_function:  # SELU
    return np.maximum(_lambda * _alpha_selu * (np.exp(x) - 1), _lambda * x)
  if 5 == num_function:  # Sigmoide
    return sigmoid(x)
  else:
    return None


# Derivatives of the activation funciton
def deriva_act(num_function, x):
  if 0 == num_function:  # Linear
    return np.ones_like(x)
  if 1 == num_function:  # Relu
    return np.greater(x, 0).astype(float)
  if 2 == num_function:  # L-Relu
    return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.01, lambda e: 1])
  if 3 == num_function:  # ELU
    return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.1 * np.exp(e), lambda e: 1])
  if 4 == num_function:  # SELU
    return np.piecewise(x, [x <= 0, x > 0], [lambda e: _lambda * _alpha_selu * np.exp(e), lambda e: _lambda])
  if 5 == num_function:  # Sigmoide
    return dev_sigmoid(x)
  else:
    return None


def sigmoid(x):
  f_x = 1 / (1 + np.exp(-x))
  return f_x


def dev_sigmoid(x):
  f_x = sigmoid(x)
  return f_x * (1 - f_x)


# Feed-Backward of SNN
def gradW(ann, param, e):
  L = ann['L']
  a = ann['a']
  z = ann['z']
  w = ann['W']
  dl = [None] * (L + 1)
  de_dw = [None] * (L + 1)

  dl[L] = e * deriva_act(SIGMOID_N_FUN, z[L])
  de_dw[L] = dl[L] @ a[L - 1].T

  for l in range(L - 1, 0, -1):
    dl[l] = (w[l + 1].T @ dl[l + 1]) * deriva_act(param['g_fun'], z[l])
    de_dw[l] = dl[l] @ a[l - 1].T

  return de_dw


# Update W and V
def updWV_rmsprop(ann, param, dE_dW, V):
  L = ann['L']
  beta = 0.9
  mu = param['mu']
  W = ann['W']
  epsilon = 0.00000001

  for l in range(1, L + 1):
    V[l] = beta * V[l] + (1 - beta) * (dE_dW[l] ** 2)
    gRMS = (1 / np.sqrt(V[l] + epsilon)) * dE_dW[l]
    W[l] = W[l] - mu * gRMS

  return W, V


def compute_Pinv(ann, H, param_ae):
  n = ann['hidden_nodes'][0]
  C = param_ae['p_inv_param']
  A = (H @ H.T) + (np.identity(n) / C)
  U, S, Vt = np.linalg.svd(A, full_matrices=False)
  S_1 = np.diag(1 / S)
  A_1 = Vt @ S_1 @ U.T
  return A_1


def updPinv(ann, xe, param_ae):
  H = ann['a'][1]
  A_1 = compute_Pinv(ann, H, param_ae)
  W_new = xe @ H.T @ A_1
  return W_new


def sort_data_random(x, y, D):
  data = np.concatenate((x, y)).T
  np.random.shuffle(data)
  data = data.T

  input_data = data[:D]
  output_data = data[D:]

  return input_data, output_data


# Get MSE
def get_mse(y_pred, y_true):
  N = y_true.shape[1]
  e = y_pred - y_true
  mse = (np.sum(e ** 2) / (2 * N))

  return mse


def get_one_hot(y, K):
  res = np.eye(K)[(y - 1).reshape(-1)]
  return res.reshape(list(y.shape) + [K]).astype(int)


# Function in charge of calculating the precision
def precision(i, cm, k):
  suma = np.sum(cm[i])

  if (suma > 0):
    prec = cm[i][i] / suma
  else:
    prec = 0

  return prec


# Function in charge of calculating the recall
def recall(j, cm, k):
  suma = np.sum(cm[:, j])

  if (suma > 0):
    rec = cm[j][j] / suma
  else:
    rec = 0

  return rec


# Function in charge of calculating the fscore
def fscore(j, cm, k):
  numerator = precision(j, cm, k) * recall(j, cm, k)
  denominator = precision(j, cm, k) + recall(j, cm, k)

  if 0 == denominator:
    return 0

  fscore = 2 * (numerator / denominator)
  return fscore
