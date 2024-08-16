

import json
import os 
from networkx import information_centrality
system_prompt={"role":"system","content":"You are a helpful assistant"}

def get_user_message(information):
    return {"role":"user","content":information}

def get_llm_message(information):
    return {"role":"assistant","content":information}
src_dir=''
target_dir=''


if not os.path.exists(target_dir):
    os.mkdir(target_dir)
file_list=os.listdir(src_dir)

for file in file_list:
    src_file=os.path.join(src_dir,file)
    target_file=os.path.join(target_dir,file)
    data_list=[]
    with open(src_file,'r') as f:
        data=json.load(f)
        for item in data:
            information={}
            information["type"]="chatml"
            message=[]
            message.append(system_prompt)
            message.append(get_user_message(item['question']))
            message.append(get_llm_message(item['answer']))
            information['messages']=message
            data_list.append(information)


    with open(target_file+'l','w') as f:
        for item in data_list:
            f.write(json.dumps(item)+'\n')
