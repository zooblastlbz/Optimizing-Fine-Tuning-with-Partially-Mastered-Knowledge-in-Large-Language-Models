import json
import os

#classify the known type by the output of the check_known.py

dir=''

target_dir=''
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

fully_known=[]
highly_known=[]
weakly_known=[]
unknown=[]
for filename in os.listdir(dir):
    file=os.path.join(dir,filename)
    with open(file,'r') as f:
        data_list=json.load(f)
    for data in data_list:
        if len(data["answers"])==1:
            tmp=dict()
            tmp['source']=filename
            tmp["question"]=data["question"]
            tmp["answer"]=data["answers"][0]
            if data['P_without_sample']==1:
                fully_known.append(tmp)
            elif data['P_without_sample'] != 0:
                tmp["P_without_sample"]=data['P_without_sample']
                highly_known.append(tmp)
            elif data['P_with_sample'] !=0 :
                tmp["P_with_sample"]=data['P_with_sample']
                weakly_known.append(tmp)
            else:
                unknown.append(tmp)
print(len(fully_known))
with open(os.path.join(target_dir,"fully_known.json"),'w') as f:
    json.dump(fully_known,f,indent=4,sort_keys=True)

with open(os.path.join(target_dir,"highly_known.json"),'w') as f:
    json.dump(highly_known,f,indent=4,sort_keys=True)

with open(os.path.join(target_dir,"weakly_known.json"),'w') as f:
    json.dump(weakly_known,f,indent=4,sort_keys=True)

with open(os.path.join(target_dir,"unknown.json"),'w') as f:
    json.dump(unknown,f,indent=4,sort_keys=True)

