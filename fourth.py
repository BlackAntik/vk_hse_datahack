import pandas as pd


US_NUM = 5000000
sit_num = len(pd.read_csv('data/nnm.csv')) 
train_users = pd.read_csv('data/train_users.csv')
users = {}
req = pd.read_csv('data/all_data.csv')
sites = req.groupby("domain_id")
for ky in sites:
    us = list(ky[1]["user_id"])
    path_id = list(ky[1]["path"])
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            par = tuple(sorted([us[i], us[j]]))
            if users.get(par) is None:
                users[par] = 0
            users[par] += 1
            if path_id[i] == path_id[j]:
                users[par] += 5

DIST = 50
pepdist_clust = [i for i in range(US_NUM)]
distclust = [[i] for i in range(US_NUM)]
for ed in users:
    v = pepdist_clust[ed[0]]
    u = pepdist_clust[ed[1]]
    if v == u:
        continue
    big_ed = 0
    small_ed = 0
    for i in distclust[v]:
        for j in distclust[u]:
            if (users.get(tuple(sorted([i, j]))) is None) or users[tuple(sorted([i, j]))] < DIST:
                small_ed += 1
            else:
                big_ed += 1
    if big_ed >= 0.7 * (big_ed + small_ed):
        if len(distclust[v]) < len(distclust[u]):
            v, u = u, v
        while len(distclust[u]) > 0:
            w = distclust[u].pop()
            pepdist_clust[w] = v
            distclust[v].append(w)
            
pepus = req.groupby("user_id")
avpep_age = [0] * US_NUM
avpep_gen = [0] * US_NUM
for i in pepus:
    avpep_age[pepdist_clust[i[0]]] += i[1]["age"][0]
    avpep_gen[pepdist_clust[i[0]]] += i[1]["gender"][0]
for i in range(US_NUM):
    if len(distclust[i]) == 0:
        continue
    avpep_age[i] /= len(distclust[i])
    avpep_gen[i] /= len(distclust[i])
pd.DataFrame({'pepdist_clust': pepdist_clust}).to_csv('data/pepdist_clust.csv', index=False)
pd.DataFrame({'avpep_age': avpep_age}).to_csv('data/avpep_age.csv', index=False)
pd.DataFrame({'avpep_gen': avpep_gen}).to_csv('data/avpep_gen.csv', index=False)