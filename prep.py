import numpy as np


# Save data: training and testing
def save_data(X, y, filename):
  np.savetxt('d' + filename, X, delimiter=',')
  np.savetxt('e' + filename, y, delimiter=',', fmt='%i')


def get_one_hot(y, K):
  res = np.eye(K)[(y - 1).reshape(-1)]
  return res.reshape(list(y.shape) + [K]).astype(int)


# Binary label
def load_data_csv(filename):
  data = np.loadtxt(filename, dtype=float, delimiter=',').T

  labels = data[-1].astype(int)
  nC = labels.max()

  X = data[:-1]
  y = get_one_hot(labels, nC)

  return X, y


# Beginning ...
def main():
  TRN_FILENAME = 'train.csv'
  TST_FILENAME = 'test.csv'

  X_trn, y_trn = load_data_csv(TRN_FILENAME)
  X_tst, y_tst = load_data_csv(TST_FILENAME)

  save_data(X_trn, y_trn, 'trn.csv')
  save_data(X_tst, y_tst, 'tst.csv')


if __name__ == '__main__':
  main()
