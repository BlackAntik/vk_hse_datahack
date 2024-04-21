import pandas as pd

all_data = pd.read_parquet('data/requests/part_0.parquet')

for i in range(0, 30):
    if 2 < i < 9:
        continue
    req = pd.read_parquet('data/requests/part_'+str(i)+'.parquet')
    all_data = pd.concat([all_data, req])
    
all_data.to_csv('data/all_data.csv')
