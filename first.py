import pandas as pd, re

def bad_index2(s):
    nums = re.findall(r'\d+', s)
    return len(nums) != 2


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

users = pd.read_csv('data/train_users.csv')
ids_set = set(users['user_id'])
users = users.set_index('user_id')

nnm = {}
sit_num = 0
for i in range(30):
    if 2 < i < 10:
        continue
    req = pd.read_parquet('data/requests/part_'+str(i)+'.parquet')  
    #req = req[:10]
    
    bad_indexes = req[req.apply(bad_index, axis=1)].index
    req = req.drop(bad_indexes, axis=0)
    req = req.reset_index()
    
    req = req.drop('index', axis=1)
    req['domain_id'] = req['referer'].apply(get_domain)
    req['path'] = req['referer'].apply(get_path)
    req['age'] = list(users['age'].loc[list(req['user_id'])])
    req['gender'] = list(users['gender'].loc[list(req['user_id'])])
    del req['referer']
    del req['user_agent']
    #req.to_parquet('data/requests/test.parquet')
    req.to_parquet('data/requests/part_'+str(i)+'.parquet')
    print("DONE "+ str(i))
    
nnm_out = pd.DataFrame({'referer' : nnm.keys(), 'site_id' : nnm.values()}).to_csv('data/nnm.csv', index=False)