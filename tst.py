import numpy as np
import utility as ut


def save_measure(cm, Fsc):
  np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')
  np.savetxt("fscores.csv", np.array(Fsc))


def load_w_dl(L):
  pesos_ae = np.load('w_ae.npz', allow_pickle=True)
  W = [None] * (L + 1)

  for i in range(1, L):
    W[i] = pesos_ae[f'arr_{i}']

  pesos_sf = np.load('w_sf.npz')
  W[L] = pesos_sf['arr_0']

  return W


def load_data_tst():
  FILE_X = 'dtst.csv'
  FILE_Y = 'etst.csv'
  X_test, y_test = ut.load_data(FILE_X, FILE_Y)
  return X_test, y_test


# Measure
def metricas(a, y):
  cm = confusion_matrix(a, y)
  k = cm.shape[0]
  fscore_result = [0] * (k + 1)

  for j in range(k):
    fscore_result[j] = ut.fscore(j, cm, k)

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


def create_ae_softmax_nn(param_ae):
  param_ae['ae_nodes'] = param_ae['ae_nodes'][:param_ae['ae_n_layers']]
  hidden_nodes = param_ae['ae_nodes']
  ann = ut.create_ann(hidden_nodes)

  return ann


# Beginning ...
def main():
  param_ae = ut.load_cnf_ae()
  xv, yv = load_data_tst()
  ann = create_ae_softmax_nn(param_ae)
  ann['W'] = load_w_dl(ann['L'])
  aL = ut.get_one_hot(np.argmax(ut.forward(ann, param_ae, xv), axis=0) + 1, yv.shape[0]).T
  cm, Fsc = metricas(aL, yv)
  print(Fsc[-1]*100)
  save_measure(cm, Fsc)


if __name__ == '__main__':   
  main()

