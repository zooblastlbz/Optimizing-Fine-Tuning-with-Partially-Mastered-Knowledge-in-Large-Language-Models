from dataset import QAdataset
import os,json
import random

#build fixed prompt for test acc
source_dir=''
target_dir=''

if not os.path.exists(target_dir):
    os.mkdir(target_dir)
    
file_list=os.listdir(source_dir)

for file in file_list:
    source_path=os.path.join(source_dir,file)
    target_path=os.path.join(target_dir,file)

    dataset=QAdataset(source_path,4)

    data_list=[]
    random.seed(42)
    for data in dataset:
        tmp={}
        tmp['message']=data[0]
        tmp['ground_truth']=data[1]
        data_list.append(tmp)

    with open(target_path,'w') as f:
        json.dump(data_list,f,indent=4,sort_keys=True)