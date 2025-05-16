import os
import json
import argparse
from tqdm import tqdm
from apis.open_router import OpenRouter

open_router_api = OpenRouter()

DETECT_PROMPT = '''
### Instructions:
1. Compare the **answer** and the **ground truth**.
2. Ignore any extra information in the answer beyond the option letter.
3. Only consider the **option letter** (e.g., 'A', 'B', 'C', 'D') when determining if the answer matches the ground truth.

### Input:
Answer: [answer]  
Ground truth: [ground_truth]  

### Task:  
Does the option letter in the answer match the ground truth?  

- Respond **only** with 'true' if the option letter matches.  
- Respond **only** with 'false' if the option letter does not match.  

### Examples:  
1. Answer: "A: The capital of France is Paris."  
    Ground truth: "A"  
    Response: "true"  

2. Answer: "C: Incorrect option."  
    Ground truth: "A"  
    Response: "false"

Now process the following input:
Answer: {}
Ground truth: {}
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", type=str, nargs='*', required=True, help="Single model name or list of model names")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    json_dir = './data/results'
    
    args = parse_args()
    jsons = args.jsons 
    
    if jsons == ['all']:
        json_files = os.listdir(json_dir)
    else:
        json_files = jsons

    acc_dict = {}
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        data = json.load(open(json_path, 'r'))
        
        model_name = os.path.splitext(json_file)[0]
        
        acc_num, acc_true = 0, 0
        for ind in tqdm(range(len(data))):
            predict = data[ind]['predict']
            answer_option = data[ind]['answer_option']
            answer = data[ind]['answer']
            
            resp = None
            if len(predict) == 1:
                resp = 'false'
                if predict.lower() == answer_option.lower():
                    acc_true += 1
                    resp = 'true'
            else:
                question = DETECT_PROMPT.format(predict, answer_option + ': ' + answer)
                
                for _ in range(5):
                    try:    
                        resp = open_router_api.infer_text(prompt=question)
                        assert (resp.lower() == 'true' or resp.lower() == 'false')
                        break
                    except Exception as e:
                        print (e)
                        continue
                
                assert (resp is not None)
                if resp.lower() == 'true':
                    acc_true += 1
                    
            acc_num += 1
            
            data[ind]['check'] = resp.lower()
            
        acc = round(1.0 * acc_true / acc_num, 4)
        acc_dict[model_name] = acc
        
        out_json_path = './data/results/{}_check.json'.format(model_name)
        with open(out_json_path, "w") as file:
            json.dump(data, file, indent=4)
        
    print (acc_dict)