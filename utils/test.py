import numpy as np
a = np.array([1,2])
b = np.array([4,5,6])
print(np.expand_dims(a,1)*np.expand_dims(b,0))

aa = np.array([1,2,3,4,5,6])
print(aa[[0,1,2]])