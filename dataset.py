
from email import message
from os import system
from torch import rand
from torch.utils.data import Dataset
import json
import random

system_prompt={"role":"system","content":"You are a helpful assistant"}

def get_user_message(information):
    return {"role":"user","content":information}

def get_llm_message(information):
    return {"role":"assistant","content":information}

def get_question_example(question,answer,number):
    return f"example {number}: Question: {question} Answer: {answer}"
def get_question(question):
    return f"Questin: {question} Answer:"

#used for build template
class QAdataset(Dataset):
    def __init__(self,filename,num_example):
        self.filename=filename
        with open(self.filename,'r') as f:
            self.data=json.load(f)
        self.length=len(self.data)
        self.num_example=num_example

    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        messages=self.random_sample_prompt(index)
        messages.append(get_user_message(self.data[index]['question']))

        return messages,self.data[index]

    def random_sample_prompt(self,index):
        index_list=[i for i in range(self.length) if i != index]
        select_sample=random.sample(index_list,self.num_example)
        messages=[]
        messages.append(system_prompt)
        for i in select_sample:
            messages.append(get_user_message(self.data[i]['question']))
            messages.append(get_llm_message(self.data[i]['answers'][0]))
        
        return messages


#used for test acc with a fixed template
class Fix_QAdataset(Dataset):
    def __init__(self,filename):
        self.filename=filename
        with open(self.filename,'r') as f:
            self.data=json.load(f)
        self.length=len(self.data)
    def __len__(self):
        return self.length
    
    def __getitem__(self, index) :
        return self.data[index]['message'],self.data[index]['ground_truth']

#used for check known type
class Known_Check_dataset(Dataset):
    def __init__(self,filename,num_example,num_sample):
        self.filename=filename
        with open(self.filename,'r') as f:
            self.data=json.load(f)
        self.length=len(self.data)
        self.num_example=num_example
        self.num_sample=num_sample

    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        batch_data=[]
        for i in range(self.num_sample):
            messages=self.random_sample_prompt(index)
            messages.append(get_user_message(self.data[index]['question']))
            batch_data.append(messages)
        return batch_data,self.data[index]


    def random_sample_prompt(self,index):
        index_list=[i for i in range(self.length) if i != index]
        select_sample=random.sample(index_list,self.num_example)
        messages=[]
        messages.append(system_prompt)
        for i in select_sample:
            messages.append(get_user_message(self.data[i]['question']))
            messages.append(get_llm_message(self.data[i]['answers'][0]))
        
        return messages
import os

#used for recheck known type, used when check on josn file built by build_checked_dataset.py,a design flaw
class Known_ReCheck_dataset(Dataset):
    def __init__(self,dir,filename,num_example,num_sample):
        self.filename=filename
        self.dir=dir
        self.example_data={}
        file_list=os.listdir(dir)
        for file in file_list:
            with open(os.path.join(dir,file),'r') as f:
                data=json.load(f)
                self.example_data[file]=data
        with open(self.filename,'r') as f:
            self.data=json.load(f)
        self.length=len(self.data)
        self.num_example=num_example
        self.num_sample=num_sample

    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        batch_data=[]
        for i in range(self.num_sample):
            messages=self.random_sample_prompt(index)
            messages.append(get_user_message(self.data[index]['question']))
            batch_data.append(messages)
        return batch_data,self.data[index]


    def random_sample_prompt(self,index):
        source=self.data[index]['source']
        example_data=self.example_data[source]
        select_sample=[]
        while True:
            need_to_sample=self.num_example-len(select_sample)
            if need_to_sample==0:
                break
            tmp_sample=random.sample(example_data,need_to_sample)
            for item in tmp_sample:
                if item not in select_sample and item['question'] != self.data[index]['question']:
                    select_sample.append(item)
        messages=[]
        messages.append(system_prompt)
        for i in select_sample:
            messages.append(get_user_message(i['question']))
            messages.append(get_llm_message(i['answers'][0]))
        
        return messages