from doctest import Example
from gc import enable
from math import e
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from dataset import QAdataset, Known_Check_dataset,Known_ReCheck_dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
import re
import json



import torch
GPU_number=torch.cuda.device_count()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the known check with specified parameters.")
    parser.add_argument("--data_source", type=str, required=True, help="Path to the origin data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--lora_adapter_path", type=str,default=None,  help="Path to the lora adapter.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling.")
    parser.add_argument("--topk", type=int, default=40, help="Top K for sampling.")
    parser.add_argument("--topp", type=float, default=1.0, help="Top P for sampling.")
    parser.add_argument("--N_temp", type=int, default=4, help="Number of examples to use.")
    parser.add_argument("--N_ex", type=int, default=10, help="Number of examples to use.")
    parser.add_argument("--N_sample", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 这里可以添加使用这些参数的逻辑，例如：
    print(f'Data Source: {args.data_source}')
    print(f"Model Path: {args.model_path}")
    print(f"lora_adapter_path: {args.lora_adapter_path}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Top K: {args.topk}")
    print(f"Top P: {args.topp}")
    print(f"N_ex: {args.N_ex}")
    print(f"N_temp: {args.N_temp}")
    print(f"N_sample: {args.N_sample}")
    print(f"Output Path: {args.output_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side="left")
    sampling_params_highly= SamplingParams(temperature=0, top_p=1,top_k=1, repetition_penalty=1.05, max_tokens=512)
    sampling_params_lowly=SamplingParams(temperature=args.temperature,top_k=args.topk,top_p=args.topp,max_tokens=512,repetition_penalty=1.05,n=args.N_sample)
    if args.lora_adapter_path is not None:  
        llm=LLM(model=args.model_path,enable_lora=True,max_lora_rank=64)
        lora_request=LoRARequest(args.lora_adapter_path,1,args.lora_adapter_path)
    else:
        llm=LLM(model=args.model_path)
    
    example_list=[]
    data_list=[]
    changed_data_list=[]
    dataset=Known_ReCheck_dataset(args.data_source,args.dataset_path,args.N_temp,args.N_ex)
    for data in dataset:
        example_list+=[tokenizer.apply_chat_template(message, tokenize=False) for message in data[0]]
        data_list.append(data[1])
    if args.lora_adapter_path is not None:
        outputs=llm.generate(example_list,sampling_params_highly,lora_request=lora_request)
    else:
        outputs=llm.generate(example_list,sampling_params_highly)
    output_len=len(outputs)
    for i in range(0,output_len,args.N_ex):
        correct=0
        num=0
        for j in range(args.N_ex):
            answer=outputs[i+j].outputs[0].text
            if data_list[i//args.N_ex]['answer'].lower() in answer.lower():
                correct+=1
            num+=1
        tmp={}
        tmp["question"]=data_list[i//args.N_ex]["question"]
        tmp["answer"]=data_list[i//args.N_ex]["answer"]
        tmp['source']=data_list[i//args.N_ex]['source']
        tmp['P_without_sample']=correct/num
        if 'P_without_sample' in data_list[i//args.N_ex].keys():
            tmp['origin_p_without_sample']=data_list[i//args.N_ex]['P_without_sample']
        else:
            tmp['origin_p_without_sample']=0
        
        changed_data_list.append(tmp)


    #remove this code to save time, it is not necessary

    '''
    not_highly_known_list=[]
    not_highly_known_example_list=[]
    for i in range(len(changed_data_list)):
        if changed_data_list[i]['P_without_sample']==0:
            not_highly_known_list.append(i)
            not_highly_known_example_list+=example_list[i*args.N_ex:(i+1)*args.N_ex]
    if args.lora_adapter_path is not None:
        outputs=llm.generate(not_highly_known_example_list,sampling_params_lowly,lora_request=lora_request)
    else:
        outputs=llm.generate(not_highly_known_example_list,sampling_params_lowly)
    output_len=len(outputs)
    for i in range(0,output_len,args.N_ex):
        correct=0
        num=0
        for j in range(args.N_ex):
            for k in range(args.N_sample):
                answer=outputs[i+j].outputs[k].text
                if data_list[not_highly_known_list[i//args.N_ex]]['answer'].lower() in answer.lower():
                    correct+=1
                num+=1
        changed_data_list[not_highly_known_list[i//args.N_ex]]['P_with_sample']=correct/num
        if 'P_with_sample' in data_list[not_highly_known_list[i//args.N_ex]].keys():
            changed_data_list[not_highly_known_list[i//args.N_ex]]['origin_p_with_sample']=data_list[not_highly_known_list[i//args.N_ex]]['P_with_sample']
    '''
    with open(args.output_path,'w') as f:
        json.dump(changed_data_list,f,indent=4,sort_keys=True)
        

if __name__ == "__main__":
    main()



