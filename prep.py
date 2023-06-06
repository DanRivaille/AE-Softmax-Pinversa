import numpy as np
import utility as ut


# Save data: training and testing
def save_data(X, Y, param):
  x_train, y_train, x_test, y_test = create_dtrn_dtst(X, Y, param['n_frame'], param['n_classes'], param['p_train'])

  np.savetxt('dtrn.csv', x_train, delimiter=',')
  np.savetxt('etrn.csv', y_train, delimiter=',', fmt='%i')

  np.savetxt('dtst.csv', x_test, delimiter=',')
  np.savetxt('etst.csv', y_test, delimiter=',', fmt='%i')


def create_dtrn_dtst(input, output, nFrame, n_classes, p_train):
  N = output.shape[0]
  nbrVariables = N // (n_classes * nFrame)
  range_class = nFrame * nbrVariables
  index_cut = int(range_class * p_train)
  data = np.concatenate((input, output.reshape(-1, 1)), axis=1)

  train_data = np.array([])
  test_data = np.array([])

  for i in range(n_classes):
    data_class = data[i * range_class:(i + 1) * range_class]

    train_class = data_class[:index_cut]
    test_class = data_class[index_cut:]

    train_data = stack_arrays(train_data, train_class)
    test_data = stack_arrays(test_data, test_class)

  np.random.shuffle(train_data)
  np.random.shuffle(test_data)

  x_train = train_data[:, :-1]
  y_train = ut.get_one_hot(train_data[:, -1].astype(int), n_classes)

  x_test = test_data[:, :-1]
  y_test = ut.get_one_hot(test_data[:, -1].astype(int), n_classes)

  return x_train.T, y_train, x_test.T, y_test


# normalize data
def data_norm(X):
  a = 0.01
  b = 0.99
  D = X.shape[1]
  for i in range(D):
    X[:, i] = normalize_var(X[:, i], a, b)

  return X


def normalize_var(x, a=0.01, b=0.99):
  x_min = x.min()
  x_max = x.max()
  if x_max > x_min:
    x = ((x - x_min) / (x_max - x_min)) * (b - a) + a
  else:
    x = a
  return x


# Obtain j-th variables of the i-th class
def data_class(x, j, i):
  return x[i, j]


def stack_arrays(stacked_array, new_array):
  if stacked_array.shape[0] != 0:
    return np.concatenate((stacked_array, new_array))
  else:
    return new_array


# Binary Label
def binary_label(class_i, N):
  labels_class = np.repeat(class_i, N)
  return labels_class


def compute_fourier_features(x, nFrame, lFrame):
  F = np.empty((nFrame, lFrame // 2))

  for n in range(nFrame):
    lower_bound = n * lFrame
    upper_bound = (n + 1) * lFrame
    current_frame = x[lower_bound:upper_bound]

    fourier = np.abs(np.fft.fft(current_frame))
    F[n] = fourier[:lFrame // 2]

  return F


# Create Features from Data
def create_features(data, param):
  nbrClass = param['n_classes']
  nbrVariables = data.shape[1]

  Y = np.array([])
  X = np.array([])

  for i in range(nbrClass):
    print('Processing class', i + 1)
    for j in range(nbrVariables):
      x = data_class(data, j, i)
      fourier_features = compute_fourier_features(x, param['n_frame'], param['l_frame'])
      X = stack_arrays(X, fourier_features)

    label = binary_label(i + 1, param['n_frame'] * nbrVariables)
    Y = stack_arrays(Y, label)

  return X, Y

def load_data(n_classes):
  list_classes = []
  for i in range(n_classes):
    file_data = f'DATA/class{i + 1}.csv'
    x = np.loadtxt(file_data, delimiter=',').T
    list_classes.append(x)

  raw_data = np.array(list_classes)
  return raw_data


# Beginning ...
def main():
  param_ae = ut.load_cnf_ae()
  raw_data = load_data(param_ae['n_classes'])
  X, Y = create_features(raw_data, param_ae)
  X = data_norm(X)
  save_data(X, Y, param_ae)


if __name__ == '__main__':
  main()
