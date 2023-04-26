# My Utility : auxiliars functions

import numpy as np

# Constantes
_alpha_elu = 0.1
_alpha_selu = 1.6732
_lambda = 1.0507
SIGMOID_N_FUN = 5


# load parameters to train the SNN
def load_cnf():
  FILE_CNF = 'cnf.csv'
  param = dict()
  cnf_list = np.loadtxt(FILE_CNF, dtype=float)

  param['n_classes'] = int(cnf_list[0])
  param['n_frame'] = int(cnf_list[1])
  param['l_frame'] = int(cnf_list[2])
  param['j_desc'] = int(cnf_list[3])

  hidden_nodes_layer_1 = int(cnf_list[4])
  hidden_nodes_layer_2 = int(cnf_list[5])

  param['g_fun'] = int(cnf_list[6])
  param['p_train'] = cnf_list[7]
  param['M_batch'] = int(cnf_list[8])
  param['mu'] = cnf_list[9]
  param['beta'] = cnf_list[10]
  param['max_iter'] = int(cnf_list[11])

  if hidden_nodes_layer_2 != 0:
    param['hidden_nodes'] = [hidden_nodes_layer_1, hidden_nodes_layer_2]
  else:
    param['hidden_nodes'] = [hidden_nodes_layer_1]

  return param


def get_one_hot(y, K):
  res = np.eye(K)[(y - 1).reshape(-1)]
  return res.reshape(list(y.shape) + [K]).astype(int)


# Load data from filename
def load_data(file_x, file_y, m):
  X = np.loadtxt(file_x, delimiter=',', dtype=float).T
  y = get_one_hot(np.loadtxt(file_y, dtype=int), m).T
  return X, y


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
def create_ann(hidden_nodes, X):
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
def updWV_sgdm(ann, param, dE_dW, V):
  L = ann['L']
  beta = param['beta']
  mu = param['mu']
  W = ann['W']

  for l in range(1, L + 1):
    V[l] = beta * V[l] + mu * dE_dW[l]
    W[l] = W[l] - V[l]

  return W, V

def sort_data_random(x, y, D):
  data = np.concatenate((x, y)).T
  np.random.shuffle(data)
  data = data.T

  input = data[:D]
  output = data[D:]

  return input, output

# Get MSE
def get_mse(y_pred, y_true):
  N = y_true.shape[1]
  e = y_pred - y_true
  mse = (np.sum(e ** 2) / (2 * N))

  return mse


# Measure
def metricas(a, y):
  cm = confusion_matrix(a, y)
  k = cm.shape[0]
  fscore_result = [0] * (k + 1)

  for j in range(k):
    fscore_result[j] = fscore(j, cm, k)

  fscore_result[k] = np.mean(fscore_result[:-1])

  return cm, fscore_result


# Confusion matrix
def confusion_matrix(a, y):
  k, N = y.shape
  cm = np.zeros((k, k), dtype=int)

  for i in range(k):
    for j in range(k):
      for n in range(N):
        if y[j, n] == 1 and a[i, n] == 1:
          cm[i, j] += 1

  return cm


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
