import numpy as np
import utility as ut

# Save Data from  Hankel's features
def save_data(X, Y, param):
  x_train, y_train, x_test, y_test = create_dtrn_dtst(X, Y, param['n_frame'], param['n_classes'], param['p_train'])

  np.savetxt('dtrn.csv', x_train, delimiter=',')
  np.savetxt('etrn.csv', y_train)

  np.savetxt('dtst.csv', x_test, delimiter=',')
  np.savetxt('etst.csv', y_test)


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
  y_train = train_data[:, -1]

  x_test = test_data[:, :-1]
  y_test = test_data[:, -1]

  return x_train, y_train, x_test, y_test


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


# E(x)
def calculate_entropy(x, N, Ix):
  interval_range, x_min, x_max = calculate_range(x, Ix)

  cross_entropy = 0
  for i in range(Ix):
    lower_bound = x_min + interval_range * i
    upper_bound = lower_bound + interval_range

    # Se obtienen la cantidad de muestras dentro del intervalo actual
    ni = np.where(np.logical_and(x >= lower_bound, x < upper_bound))[0].shape[0]

    if ni != 0:
      pi = ni / N

      entropy = pi * np.log2(pi)
      cross_entropy += entropy

  return -cross_entropy


def calculate_range(x, Ix, x_min=0.01, x_max=1):
  range = (x_max - x_min) / Ix
  return range, x_min, x_max


# Binary Label
def binary_label(class_i, N):
  labels_class = np.repeat(class_i, N)
  return labels_class


def compute_fourier(X):
  return np.abs(np.fft.fft(X))


# Fourier spectral entropy
def entropy_spectral(component):
  N = component.shape[0]
  Ix = int(np.sqrt(N))

  c_fourier = compute_fourier(component)
  c_fourier_normalized = normalize_var(c_fourier)
  return calculate_entropy(c_fourier_normalized, N, Ix)


def extract_dyadic_component(H):
  a = np.concatenate((H[0], H[1, -1:]))
  b = np.concatenate((H[0, :1], H[1]))
  c = (a + b) / 2
  return c


def create_hankel_matrix(signal, L):
  N = signal.shape[0]
  K = N - L + 1
  hankel = np.empty((L, K))
  for i in range(L):
    hankel[i] = signal[i:i + K]

  return hankel


def get_H(U, S, Vt, i):
  Hi = S[i] * (U[:, i].reshape(-1, 1) @ Vt[i, :].reshape(1, -1))
  return Hi


def decompose_matrix(U, S, Vt):
  H1 = get_H(U, S, Vt, 0)
  H2 = get_H(U, S, Vt, 1)
  return H1, H2


def svd(H):
  U, S, Vt = np.linalg.svd(H, full_matrices=False)
  return U, S, Vt


def compute_next_level(H, max_level, current_level, C, S_components):
  U, S, Vt = svd(H)
  H1, H2 = decompose_matrix(U, S, Vt)

  if max_level == current_level:
    C1 = extract_dyadic_component(H1)
    C2 = extract_dyadic_component(H2)

    C.append(C1)
    C.append(C2)
    S_components.append(S[0])
    S_components.append(S[1])
  else:
    compute_next_level(H1, max_level, current_level + 1, C, S_components)
    compute_next_level(H2, max_level, current_level + 1, C, S_components)


# Hankel-SVD
def hankel_svd(frame, j):
  C = []
  Sc = []
  H_matrix = create_hankel_matrix(frame, 2)
  compute_next_level(H_matrix, j, 1, C, Sc)

  return C, Sc


# Hankel's features 
def hankel_features(x, nFrame, lFrame, j):
  F = np.empty((nFrame, 2 ** (j + 1)))
  for n in range(nFrame):
    lower_bound = n * lFrame
    upper_bound = (n + 1) * lFrame
    current_frame = x[lower_bound:upper_bound]
    C, Sc = hankel_svd(current_frame, j)

    entropy_c = []
    for c in C:
      spectral_entropy = entropy_spectral(c)
      entropy_c.append(spectral_entropy)

    F[n] = np.concatenate((entropy_c, Sc))

  return F


# Obtain j-th variables of the i-th class
def data_class(x, j, i):
  return x[i, j]


def stack_arrays(stacked_array, new_array):
  if stacked_array.shape[0] != 0:
    return np.concatenate((stacked_array, new_array))
  else:
    return new_array


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
      F = hankel_features(x, param['n_frame'], param['l_frame'], param['j_desc'])
      X = stack_arrays(X, F)

    label = binary_label(i + 1, param['n_frame'] * nbrVariables)
    Y = stack_arrays(Y, label)

  return X, Y


# Load data from ClassXX.csv
def load_data(n_classes):
  list_classes = []
  for i in range(n_classes):
    file_data = f'data/Data{n_classes}/class{i + 1}.csv'
    x = np.loadtxt(file_data, delimiter=',').T
    list_classes.append(x)

  raw_data = np.array(list_classes)
  return raw_data


# Beginning ...
def main():
  param = ut.load_cnf()
  data = load_data(param['n_classes'])
  X, Y = create_features(data, param)
  X = data_norm(X)
  save_data(X, Y, param)


if __name__ == '__main__':
  main()
