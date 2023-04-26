import numpy as np
import utility as ut

def save_measure(cm, Fsc):
  np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')
  np.savetxt("fscores.csv", np.array(Fsc))


def load_w(L):
  pesos = np.load('w_snn.npz', allow_pickle=True)
  W = [None] * (L + 1)

  for i in range(1, L + 1):
    W[i] = pesos[f'arr_{i}']

  return W


def load_data_test(param):
  FILE_X = 'dtst.csv'
  FILE_Y = 'etst.csv'
  X_test, y_test = ut.load_data(FILE_X, FILE_Y, param['n_classes'])
  return X_test, y_test
    

# Beginning ...
def main():			
  param = ut.load_cnf()
  xv, yv  = load_data_test(param)
  ann = ut.create_ann(param['hidden_nodes'], xv)
  ann['W'] = load_w(ann['L'])
  aL = ut.get_one_hot(np.argmax(ut.forward(ann, param, xv), axis=0) + 1, param['n_classes']).T
  cm, Fsc = ut.metricas(aL, yv)
  save_measure(cm, Fsc)


if __name__ == '__main__':   
  main()

