import os
import IO
from rl_learning import PPBP
import random

files = 8
t = 100 
size = 40
B, _ = PPBP.createPPBPTrafficGen(t/0.1, 5, 0.8, size/12.5, 1, 0.000125 * 10)

for f in range(files):
    randint = random.randint(100, len(B))
    path = "data" + str(f) + ".csv"
    fp1 = IO.open_for_writing(path)
    for i in range(len(B)):
        idx = (randint + i) % len(B)
        fp1.write( str(B[idx]) + '\n')

    IO.close_for_writing(fp1)
