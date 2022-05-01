import numpy as np
from numpy import load

data = load('./data/mnist.npz')
for key, value in data.items():

    np.savetxt("./somepath" + key + ".csv", value)