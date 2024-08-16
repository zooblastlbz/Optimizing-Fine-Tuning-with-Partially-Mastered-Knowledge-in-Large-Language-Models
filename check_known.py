from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from dataset import QAdataset, Known_Check_dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
import re
import json
device = "cuda" # the device to load the model onto


#check known type for a single josn file

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("path to model",padding_side="left")

# Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params_highly= SamplingParams(temperature=0, top_p=1,top_k=1, repetition_penalty=1.05, max_tokens=512)

sampling_params_lowly=SamplingParams(temperature=0.5,top_k=40,top_p=1,max_tokens=512,repetition_penalty=1.05,n=16)
# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="path to model",device=device)

# Prepare your pro


def check_konwn(filename,T=0.5,N_ex=10,N_sample=16,top_k=40):
    checked_data_list=[]
    example_list=[]
    data_list=[]
    dataset=Known_Check_dataset(filename,4,N_ex)
    for data in dataset:
        example_list+=[tokenizer.apply_chat_template(message, tokenize=False) for message in data[0]]
        data_list.append(data[1])
    outputs=llm.generate(example_list,sampling_params_highly)

    #batch_generated_ids = model.generate(batch_model_inputs.input_ids,attention_mask=batch_model_inputs.attention_mask,max_new_tokens=512,do_sample=False,top_k=None,top_p=None,temperature=None)
    #batch_responses = [tokenizer.decode(generated_ids, skip_special_tokens=True) for generated_ids in batch_generated_ids]

    output_len=len(outputs)
    for i in range(0,output_len,N_ex):
        correct=0
        num=0
        for j in range(N_ex):
            answer=outputs[i+j].outputs[0].text
            if data_list[i//N_ex]['answers'][0].lower() in answer.lower():
                correct+=1
            num+=1
        tmp={}
        tmp["question"]=data_list[i//N_ex]["question"]
        tmp["answers"]=data_list[i//N_ex]["answers"]
        tmp['P_without_sample']=correct/num

        checked_data_list.append(tmp)


    not_highly_known_list=[]
    not_highly_known_example_list=[]
    for i in range(len(checked_data_list)):
        if checked_data_list[i]['P_without_sample']==0:
            not_highly_known_list.append(i)
            not_highly_known_example_list+=example_list[i*N_ex:(i+1)*N_ex]

    outputs=llm.generate(not_highly_known_example_list,sampling_params_lowly)
    output_len=len(outputs)
    for i in range(0,output_len,N_ex):
        correct=0
        num=0
        for j in range(N_ex):
            for k in range(N_sample):
                answer=outputs[i+j].outputs[k].text
                if data_list[not_highly_known_list[i//N_ex]]['answers'][0].lower() in answer.lower():
                    correct+=1
                num+=1
        checked_data_list[not_highly_known_list[i//N_ex]]['P_with_sample']=correct/num
        

    if not os.path.exists('output path'):
        os.mkdir('output path')
    with  open(os.path.join('output path',filename.split('/')[-1]),'w') as f:
        json.dump(checked_data_list, f, indent=4, sort_keys=True)
        


import sys

if __name__ == "__main__":
    file=sys.argv[1]    
    check_konwn("dataset dir"+file)









