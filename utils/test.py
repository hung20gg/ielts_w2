import numpy as np
from sklearn.mixture import GaussianMixture
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
gm = GaussianMixture(n_components=4, random_state=0).fit(X)
gm.means_

gm.predict([[0, 0], [12, 3]])
prob = gm.predict_proba([[0, 0], [12, 3]])
print(prob)
print(np.linalg.norm(prob, axis=1))