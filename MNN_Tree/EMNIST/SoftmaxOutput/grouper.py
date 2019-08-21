import torch
a = torch.load("emnist_root_softmax_output.pth")
print("root:")

l = []
for i in range(47):
    l.append([i])
    for j in range(i,47):
        if a[i][0][j] >= 1/47:
            if i != j:
                l[i].append(j)


from collections import defaultdict
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
#print(final_dict)


seen = []
group_root = []
for i in final_dict:
    if i not in seen:
        group_root.append(final_dict[i])
        for j in final_dict[i]:
            seen.append(j)
for i in group_root:
    print(i)


print("\n\n0:")
mapper = dict(zip(list(range(0,len(group_root[0]))), group_root[0]))
a = torch.load('emnist_0_softmax_output.pth')
l = []
for i in range(12):
    l.append([i])
    for j in range(0,12):
        if a[i][0][j] >= 1/12:
            if i != j:
                l[i].append(j)


from collections import defaultdict
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
#print(final_dict)


seen = []
group = []
for i in final_dict:
    if i not in seen:
        group.append(final_dict[i])
        for j in final_dict[i]:
            seen.append(j)
#print("0: ", group)
for i in group:
    x = []
    for j in i:
        x.append(mapper[j])
    print(x)


print("\n\n1:")
mapper = dict(zip(list(range(0,len(group_root[1]))), group_root[1]))
a = torch.load('emnist_1_softmax_output.pth')
l = []
for i in range(4):
    l.append([i])
    for j in range(0,4):
        if a[i][0][j] >= 1/4:
            if i != j:
                l[i].append(j)


from collections import defaultdict
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
#print(final_dict)


seen = []
group = []
for i in final_dict:
    if i not in seen:
        group.append(final_dict[i])
        for j in final_dict[i]:
            seen.append(j)
for i in group:
    x = []
    for j in i:
        x.append(mapper[j])
    print(x)


print("\n\n3:")
mapper = dict(zip(list(range(0,len(group_root[3]))), group_root[3]))
a = torch.load('emnist_3_softmax_output.pth')
l = []
for i in range(13):
    l.append([i])
    for j in range(i,13):
        if a[i][0][j] >= 1/13:
            if i != j:
                l[i].append(j)


from collections import defaultdict
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
#print(final_dict)


seen = []
group = []
for i in final_dict:
    if i not in seen:
        group.append(final_dict[i])
        for j in final_dict[i]:
            seen.append(j)


for i in group:
    x = []
    for j in i:
        x.append(mapper[j])
    print(x)


print("\n\n8:")
mapper = dict(zip(list(range(0,len(group_root[8]))), group_root[8]))
a = torch.load('emnist_8_softmax_output.pth')
l = []
for i in range(4):
    l.append([i])
    for j in range(0,4):
        if a[i][0][j] >= 1/4:
            if i != j:
                l[i].append(j)


from collections import defaultdict
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
#print(final_dict)


seen = []
group = []
for i in final_dict:
    if i not in seen:
        group.append(final_dict[i])
        for j in final_dict[i]:
            seen.append(j)
for i in group:
    x = []
    for j in i:
        x.append(mapper[j])
    print(x)
