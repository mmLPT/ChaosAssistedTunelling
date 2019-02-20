import numpy as np
import sys

ifile=sys.argv[1]

data=np.load(ifile)

for j in data.files:
	print(j," = ", data[j])
