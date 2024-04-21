import pandas as pd, pickle

sit_num = len(pd.read_csv('data/nnm.csv')) 
train_users = pd.read_csv('data/train_users.csv')
visit = [[] for i in range(sit_num)]
sites = {}

#req = pd.read_parquet('data/requests/part_0.parquet')[:1000000]
req = pd.read_csv('data/all_data.csv')
users = req.groupby("user_id")
for ky in users:
    us = list(ky[1]["domain_id"])
    path_id = list(ky[1]["path"])
    for i in range(len(us)):
        visit[us[i]].append(ky[0])
        for j in range(i + 1, len(us)):
            par = tuple(sorted([us[i], us[j]]))
            if sites.get(par) is None:
                sites[par] = 0
            sites[par] += 1
            if path_id[i] == path_id[j]:
                sites[par] += 5
            
            
with open('sites.pkl', 'wb') as f:
    pickle.dump(sites, f)

#pd.DataFrame({'edges' : sites.keys(), 'cnt' : sites.values()}).to_csv('data/graph.csv')