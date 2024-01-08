import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_file_path = 'data/lab2_train.csv' 
test_file_path = 'data/lab2_test.csv'

train_data = pd.read_csv(train_file_path, delimiter=',')
test_data = pd.read_csv(test_file_path, delimiter=',')

train_data.dropna(axis=0, inplace=True)
user_data = train_data["user_from_id"].value_counts()

valid_users = user_data[user_data >= 20].index
train_data = train_data[train_data["user_from_id"].isin(valid_users)]

train_data = train_data.drop_duplicates()


def nmf(X, mask, hyper_param, n_components: int, max_iter: int=1000, tol: float=1e-3):
  """
  Decomposes the original sparse matrix X into two matrices W and H. 
  """
  # Initialize W and H with random non-negative values
  W = np.random.rand(X.shape[0], n_components)
  H = np.random.rand(n_components, X.shape[1])

  E = np.sum(((X - W @ H)**2) * mask)
  newE = 0.
  i = 0

  x_masked = X * mask
  while E - newE > tol and i < max_iter: 
    E = np.sum(((X - W @ H)**2) * mask)

    nominatorH = W.T @ x_masked
    denominatorH = (W.T @ ((W @ H) * mask) + 1e-9)
    H = H * nominatorH / denominatorH

    nominatorW = (x_masked @ H.T)
    denominatorW = (((W @ H) * mask) @ H.T + 1e-9)
    W = W * nominatorW / denominatorW

    newE = np.sum(((X - W @ H)**2) * mask)
    i += 1

  print(E)
  print(i)
  return W, H

x = train_data.to_numpy()[:,[0,1,3]].astype(float)
y = train_data.to_numpy()[:, 2]
y = np.where(y == False, 0., 1.)
x[:,2] = np.where(x[:,2] == False, 0., 1.)
x[:,:2] = x[:,:2] / 5000.
y = y.astype(float)
x = x.astype(float)


def split_data():
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.001, random_state=42)

  mask = np.ones((len(x), 4))
  mask[len(x_train):, 3:] = 0

  train_matrix = np.hstack((np.vstack((x_train, x_test)), np.vstack((y_train.reshape(-1, 1), y_test.reshape(-1, 1)))))
  return train_matrix, y_train, mask, len(x_train)

def eval(X, M, components, train_length):
  ys = []

  W, H = nmf(X, M, 100., components)
  predicted_matrix = W @ H
  return predicted_matrix[:train_length, 3:]

def calculate():
  for j in range(35, 36, 1):
    plt.figure(figsize=(15,8))
    train_matrix, y_test, mask, train_length = split_data()
    results_gotten = eval(train_matrix, mask, j, train_length)
    results_actual = y_test
    plt.scatter(results_actual, results_gotten, label=f"{j} components", alpha=0.5)
    plt.xlabel("real score")
    plt.ylabel("predicted score")
    plt.title("real score vs predicted score")
    plt.legend()
    plt.show()

    print(f"{j} components")
    print(np.sum((results_actual - results_gotten)**2))

calculate()
  