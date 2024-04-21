import pandas as pd, re

from sklearn.metrics import mean_absolute_error as mae

def get_domain(s):
    global nnm, sit_num
    num = int(re.findall(r'\d+', s)[0])
    if nnm.get(num) is None:
        nnm[num] = sit_num
        sit_num += 1
    return nnm[num]


def get_path(s):
    nums = re.findall(r'\d+', s)
    return int(nums[1])

def bad_index(row):
    return row['user_id'] not in ids_set or len(row['referer']) == 0


new_users = pd.read_csv('data/test_users.csv')
nreq = pd.read_parquet('../data/requests/part_0.parquet')
nnm = pd.read_csv('data/nnm.csv')
site_clust = pd.read_csv('data/site_clust2.csv')
av_gen = pd.read_csv('data/av_gen2.csv')
av_age = pd.read_csv('data/av_age2.csv')
sit_num = len(site_clust)


ids_set = set(new_users['user_id'])
new_users = new_users.set_index('user_id')

#for i in range(20, 30):
for i in range(30):
    if 2 < i < 10:
        continue
    req = pd.read_parquet('../data/requests/part_'+str(i)+'.parquet')
    bad_indexes = req[req.apply(bad_index, axis=1)].index
    req = req.reset_index()
    
    req = req.drop('index', axis=1)
    req['domain_id'] = req['referer'].apply(get_domain)
    req['path'] = req['referer'].apply(get_path)
    req['age'] = list(new_users['age'].loc[list(req['user_id'])])
    req['gender'] = list(new_users['gender'].loc[list(req['user_id'])])
    del req['referer']
    del req['user_agent']

    req = req.drop(bad_indexes, axis=0)

    
    nreq = pd.concat([nreq, req])


#pepdist_clust, pep_clust, av_gen, av_age, avpep_gen, avpep_age, visit
US_NUM = 20000000
freq = [0] * sit_num
sos_clust = [0] * US_NUM
nusers = nreq.groupby("user_id")
our_ans_age = []
our_ans_gen = []
right_ans_age = []
right_ans_gen = []
for ky in nusers:
    us = list(ky[1]["domain_id"])
    mx = 0
    ind = 0
    mx1 = 0
    ind1 = 0
    for i in range(len(us)):
        freq[us[i]] += 1
        if freq[us[i]] > mx:
            mx = freq[us[i]]
            ind = us[i]
    our_ans_age.append(av_age[site_clust[ind]])
    our_ans_gen.append(av_age[site_clust[ind]])
    right_ans_age.append(ky[1]["age"][0])
    right_ans_gen.append(ky[1]["gender"][0])

print('age mean absolute error', mae(our_ans_age, right_ans_age))
print('gen mean absolute error', mae(our_ans_gen, right_ans_gen))
pd.DataFrame({'pred_age': our_ans_age}).to_csv('data/pred_age.csv')
pd.DataFrame({'pred_gen': our_ans_gen}).to_csv('data/pred_gen.csv')

pd.DataFrame({'users_id': new_users['user_id'], 'gender' : our_ans_gen, 'age': our_ans_age}).to_csv('predictions.csv', index=False)