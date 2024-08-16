from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
from dataset import Fix_QAdataset
import transformers
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the known check with specified parameters.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--lora_adapter_path", type=str,default=None,  help="Path to the lora adapter.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--N_ex", type=int, default=10, help="Number of examples to use.")
    parser.add_argument("--output_path", type=str, default='', help="Path to save the output.")
    
    args = parser.parse_args()
    return args

#test accuacry  using fix template
import logging

def test_acc_fix():
    args = parse_args()
    
    # 这里可以添加使用这些参数的逻辑，例如：
    print(f"Model Path: {args.model_path}")
    print(f"lora_adapter_path: {args.lora_adapter_path}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"N_ex: {args.N_ex}")
    print(f"Output Path: {args.output_path}")
    if args.lora_adapter_path is not None:
        log_file=os.path.join(args.output_path,args.lora_adapter_path.split('/')[-1]+'.log')
    else:
        log_file=os.path.join(args.output_path,args.model_path.split('/')[-1]+'.log')
    logging.basicConfig(filename=log_file,level=logging.INFO)
    tokenizer=transformers.AutoTokenizer.from_pretrained(args.model_path,padding_side="left")
    if args.lora_adapter_path is not None:  
        llm=LLM(model=args.model_path,enable_lora=True,max_lora_rank=64)
        lora_request=LoRARequest(args.lora_adapter_path,1,args.lora_adapter_path)
    else:
        llm=LLM(model=args.model_path)
    sampling_params=SamplingParams(temperature=0,top_k=1,top_p=1,max_tokens=512,repetition_penalty=1.05,n=1)
    if os.path.isdir(args.dataset_path):
        total_num=0
        total_correct=0
        logging.info('test accuacry on dir')
        file_list=os.listdir(args.dataset_path)
        for file in file_list:
            print(os.path.join(args.dataset_path,file))
            dataset=Fix_QAdataset(os.path.join(args.dataset_path,file))
            correct=0
            batch_data=[]
            for data in dataset:
                batch_data.append(tokenizer.apply_chat_template(data[0], tokenize=False))
            if args.lora_adapter_path is not None:
                output=llm.generate(batch_data,sampling_params,lora_request=lora_request)
            else:
                output=llm.generate(batch_data,sampling_params)
            for i in range(len(output)):
                total_num+=1
                if dataset[i][1]['answers'][0] in output[i].outputs[0].text:
                    correct+=1
                    total_correct+=1

            logging.info(f'file:{file} accuracy:{correct/len(dataset)}')
        logging.info(f'total accuracy:{total_correct/total_num}')
    else:
        dataset=Fix_QAdataset(args.dataset_path)
        correct=0
        batch_data=[]
        for data in dataset:
            batch_data.append(tokenizer.apply_chat_template(data[0], tokenize=False))
        if args.lora_adapter_path is not None:
            output=llm.generate(batch_data,sampling_params,lora_request=lora_request)
        else:
            output=llm.generate(batch_data,sampling_params)
        for i in range(len(output)):
            if dataset[i][1]['answers'][0] in output[i].outputs[0].text:
                correct+=1
        logging.info(f'file:{args.dataset_path},accuracy:{correct/len(dataset)}')
    logging.info('finish test')

if __name__ == '__main__':
    test_acc_fix()