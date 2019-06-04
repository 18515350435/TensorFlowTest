# coding: utf-8
import numpy as np
a = [0,1,2,3,4,5,6]
b = [10,11,12,13,14,15,16]
print(a, b)
# result:[0 1 2 3 4 5 6 7 8 9] [10 11 12 13 14 15 16 17 18 19]
state = np.random.get_state()
np.random.shuffle(a)
np.random.set_state(state)
np.random.shuffle(b)
print(a)
print(b)

