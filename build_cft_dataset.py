import json
import os


#build cft dataset
dir=''

target_dir=''
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

dataset=[]
replay_dataset=[]

#control the component of the dataset by changing the file_list and replay_list
replay_list=[]
file_list=[]
for filename in file_list:
    file=os.path.join(dir,filename)
    with open(file,'r') as f:
        data_list=json.load(f)
    for data in data_list:

        if data['P_without_sample']>0 and data['P_without_sample']<1:
            tmp=dict()
            tmp['source']=data['source']
            tmp["question"]=data["question"]
            tmp["answer"]=data["answer"]
            tmp['P_without_sample']=data['P_without_sample']
            dataset.append(tmp)

for filename in replay_list:
    file=os.path.join(dir,filename)
    with open(file,'r') as f:
        data_list=json.load(f)
    for data in data_list:
        if data['P_without_sample']==11: 
            tmp=dict()
            tmp['source']=data['source']
            tmp["question"]=data["question"]
            tmp["answer"]=data["answer"]
            tmp['P_without_sample']=data['P_without_sample']
            replay_dataset.append(tmp)

print(len(dataset))
print(len(replay_dataset))

with open(os.path.join(target_dir,"cft_data_epoch_8.json"),'w') as f:
    json.dump(dataset,f,indent=4,sort_keys=True)

with open(os.path.join(target_dir,"replay_data_epoch_8.json"),'w') as f:
    json.dump(replay_dataset,f,indent=4,sort_keys=True)
