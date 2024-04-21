import pandas as pd, pickle

'''
sit_num = len(pd.read_csv('data/nnm.csv')) 
train_users = pd.read_csv('data/train_users.csv')
visit = [[] for i in range(sit_num)]
sites = {}

#req = pd.read_parquet('data/requests/part_0.parquet')[:1000000]
req = pd.read_csv('data/all_data_test.csv')
users = req.groupby("user_id")
for ky in users:
    us = list(ky[1]["domain_id"])
    for i in range(len(us)):
        visit[us[i]].append(ky[0])
        for j in range(i + 1, len(us)):
            par = tuple(sorted([us[i], us[j]]))
            if sites.get(par) is None:
                sites[par] = 0
            sites[par] += 1
            if ky[1]["path_id"][i] == ky[1]["path_id"][j]:
                sites[par] += 5
            
print("DONE")

'''

sit_num = len(pd.read_csv('data/nnm.csv'))
req = pd.read_csv('data/all_data.csv')

sites = {}
        
with open('sites.pkl', 'rb') as f:
    sites = pickle.load(f)

DIST = 10
site_clust = [i for i in range(sit_num)]
clust = [[i] for i in range(sit_num)]
for site in sites:
    v = site_clust[site[0]]
    u = site_clust[site[1]]
    if v == u:
        continue
    big_ed = 0
    small_ed = 0
    for i in clust[v]:
        for j in clust[u]:
            if (sites.get(tuple(sorted([i, j]))) is None) or sites[tuple(sorted([i, j]))] < DIST:
                small_ed += 1
            else:
                big_ed += 1
    if big_ed >= 0.6 * (big_ed + small_ed):
        if len(clust[v]) < len(clust[u]):
            v, u = u, v
        while len(clust[u]) > 0:
            w = clust[u].pop()
            site_clust[w] = v
            clust[v].append(w)

users = req.groupby("user_id")
pep_clust = [[] for i in range(sit_num)]
PER_IN_CL = 2
gist = [(0, i) for i in range(sit_num)]
for ind in users:
    i = list(ind[1]["domain_id"])
    us = set()
    for j in i:
        gist[site_clust[j]] = (gist[site_clust[j]][0] + 1, gist[site_clust[j]][1])
        us.add(site_clust[j])
    nw = []
    for j in us:
        nw.append(gist[j])
        gist[j] = (0, gist[j][1])
    nw.sort(reverse = True)
    for j in range(min(PER_IN_CL, len(nw))):
        pep_clust[nw[j][1]].append((list(ind[1]["age"])[0], list(ind[1]["gender"])[0]))

av_age = [0] * sit_num
av_gen = [0] * sit_num
for i in range(sit_num):
    if len(pep_clust[i]) == 0:
        continue
    for j in pep_clust[i]:
        av_age[i] += j[0]
        av_gen[i] += j[1]
    av_age[i] /= len(pep_clust[i])
    av_gen[i] /= len(pep_clust[i])

    
pd.DataFrame({'av_age': av_age}).to_csv('data/av_age2.csv', index=False)
pd.DataFrame({'av_gen': av_gen}).to_csv('data/av_gen2.csv', index=False)
pd.DataFrame({'site_clust': site_clust}).to_csv('data/site_clust2.csv', index=False)