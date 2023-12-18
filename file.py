import numpy as np
import pandas as pd


def nmf(X: pd.DataFrame, n_components: int, max_iter: int=1000, tol: float=1e-3):
  """
  Decomposes the original sparse matrix X into two matrices W and H. 
  """
  # Initialize W and H with random non-negative values
  W = np.random.rand(X.shape[0], n_components)
  H = np.random.rand(n_components, X.shape[1])

  # START ANSWER

  E = np.linalg.norm(X - (W @ H))
  newE = 0.
  i = 0

  while E - newE > tol and i < max_iter:
    E = np.linalg.norm(X - (W @ H))
    denominatorW = (W @ H @ H.T)
    denominatorW[denominatorW == 0] = 1e-9
    W = np.divide(np.multiply(W, X @ H.T), denominatorW)
    denominatorH = (W.T @ W @ H)
    denominatorH[denominatorH == 0] = 1e-9
    H = np.divide(np.multiply(H, W.T @ X), denominatorH)

    newE = np.linalg.norm(X - (W @ H))
    i += 1

  # END ANSWER

  return W, H
dataNp = np.array([[3,1,1,3,0.1], [1,2,4,1,3], [3,1,1,3,1], [4,3,5,4,4]])
dataF = pd.DataFrame(dataNp)

W, H = nmf(dataF, 4)
print(W@H)
