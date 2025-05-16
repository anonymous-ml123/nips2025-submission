import os, sys
import json
import argparse
import random
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image
from dataclasses import asdict
from datasets import load_dataset

load_dotenv()

from apis.open_router import OpenRouter
from config import MODEL_API_MAP

open_router_api = OpenRouter()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs='*', required=True, help="Single model name or list of model names")
    args = parser.parse_args()
    return args


def load_data(data_path='./data/label.json'):
    if not os.path.exists(data_path):
        print ('Begin downloading dataset...')
        dataset = load_dataset('./data/nips2025-data')

        out_image_dir = './data/images'
        if not os.path.exists(out_image_dir):
            os.makedirs(out_image_dir)
        out_json_path = './data/label.json'

        out_list = []
        for ind in tqdm(range(len(dataset['test']))):
            item = dataset['test'][ind]
            out_list.append({
                'image_id': item['file_name'],
                'category': item['category'],
                'sub_category': item['sub_category'],
                'question': item['question'],
                'choices': item['choices'],
                'answer': item['answer'],
                'answer_option': item['answer_option'],
                'prompt': item['prompt'],
            })

            out_img_path = os.path.join(out_image_dir, item['file_name'])
            if not os.path.exists(out_img_path):
                item['image'].save(out_img_path)

        with open(out_json_path, "w") as f:
            json.dump(out_list, f, indent=4)
        
    data = json.load(open(data_path, 'r'))
    return data


def eval(data=None, model=None, api_name=None, tag=None):
    if tag == 'vllm':
        from vllm import SamplingParams, LLM
        from models.vllm_infer import vllm_model_example_map
        
        modality='image'
        
        req_data = vllm_model_example_map[model_name](modality=modality, model_path=api_name)
        # Disable other modalities to save memory
        default_limits = {"image": 0, "video": 0, "audio": 0}
        req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
            req_data.engine_args.limit_mm_per_prompt or {})
        

        sampling_params = SamplingParams(temperature=0.2,
                                        max_tokens=2048,
                                        stop_token_ids=req_data.stop_token_ids)
        
        engine_args = asdict(req_data.engine_args) | {
            "disable_mm_preprocessor_cache": True,
        }
        llm = LLM(**engine_args)
    
    for ind in tqdm(range(len(data))):
        image_id = data[ind]['image_id']
        image_path = os.path.join('./data/images', image_id)
        assert os.path.exists(image_path)
        
        prompt = data[ind]['prompt']
        
        ans = ''
        for _ in range(3):
            try:
                if tag == 'open_router':
                    ans = open_router_api.infer(api_name, image_path, prompt)
                elif tag == 'qwen_vl':
                    ans = qwen_vl_api.infer(api_name, image_path, prompt)
                elif tag == 'vllm':
                    image = Image.open(image_path).convert('RGB')
                    
                    inputs = {
                        "prompt": req_data.prompts[0].format(prompt),
                        "multi_modal_data": {
                            "image": image
                        },
                    }
                    outputs = llm.generate(inputs, sampling_params=sampling_params)
                    ans = outputs[0].outputs[0].text
                break
            except Exception as e:
                print (e)
                continue
        
        data[ind]['predict'] = ans
    
    out_json_path = './data/results/{}_{}.json'.format(model, tag)
    with open(out_json_path, "w") as file:
        json.dump(data, file, indent=4)
                

if __name__ == '__main__':
    data = load_data()
    args = parse_args()
    
    models = args.model  
    
    if models == ['all']:
        for tag in MODEL_API_MAP:
            for model_name, api_name in MODEL_API_MAP[tag].items():
                eval(data=data, model=model_name, api_name=api_name, tag=tag)
    else:
        for model_name in models:
            tag = None
            for tag_temp in MODEL_API_MAP:
                if model_name in MODEL_API_MAP[tag_temp]:
                    tag = tag_temp
                    break
            if tag is not None:
                api_name = MODEL_API_MAP[tag][model_name]
                eval(data=data, model=model_name, api_name=api_name, tag=tag)
            else:
                raise ValueError('Model: {} is not supported!'.format(model_name))
