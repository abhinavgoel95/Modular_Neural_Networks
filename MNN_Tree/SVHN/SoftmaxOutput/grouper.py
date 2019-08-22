import torch
import numpy as np
import math
from collections import defaultdict

a = torch.load("svhn_root_softmax_output.pth")

def sigmoidal(m,s):
    return 1/( 1 + math.exp(-(s - (1/m)) / (1/(10*m))) )

def make_dict(l):
    final_dict = defaultdict(list)
    for i in l:
        for j in i:
            for k in i:
                if k not in final_dict[j]:
                    final_dict[j].append(k)
                if j not in final_dict[k]:
                    final_dict[k].append(j)

    for i in final_dict:
        for j in final_dict[i]:
            for k in final_dict[j]:
                if k not in final_dict[i]:
                    final_dict[i].append(k)
    return final_dict


l = []
for i in range(10):
    l.append([i])
    for j in range(0,10):
        prob = sigmoidal(10, a[i][0][j])
        if np.random.choice(2, 10000, p = [1 - prob, prob]).mean() > 0.5:
            if i != j:
                l[i].append(j)

final_dict = make_dict(l)

seen = []
group = []
for i in final_dict:
    if i not in seen:
        group.append(final_dict[i])
        for j in final_dict[i]:
            seen.append(j)
print("root: ")
for i in group:
    print(i)
