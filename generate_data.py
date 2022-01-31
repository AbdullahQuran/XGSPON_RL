import os
import IO
from rl_learning import PPBP
import random

files = 1
t = 1000 
size = 40
B, _ = PPBP.createPPBPTrafficGen(t/0.1, 5, 0.8, size/12.5, 1, 0.000125 * 10)

for f in range(files):
    randint = random.randint(100, len(B))
    path = "lstm" + str(f) + ".csv"
    fp1 = IO.open_for_writing(path)
    count = len(B)
    j = 0
    for i in range(int(count/10)):
        idx = (randint + j) % len(B)
        fp1.write(str(B[idx] *  2 * 10 * 0.62) + '\n')
        j = j + 10
    IO.close_for_writing(fp1)
