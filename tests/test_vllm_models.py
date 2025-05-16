import os
import sys
from dataclasses import asdict
from PIL import Image
from vllm import SamplingParams, LLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_API_MAP
from models.vllm_infer import vllm_model_example_map


class TestVLLMModels:
    def __init__(self):
        self.vllm_models = MODEL_API_MAP['vllm']
        self.vllm_model_example_map = vllm_model_example_map
        
        
    def test_vllm_models(self, image_path, question, modality='image'):
        image = Image.open(image_path).convert('RGB')
        
        out_json_path = './data/tests/vllm_models.json'
        ans_dict = {}
        
        for model_name, model_path in self.vllm_models.items():
            print (f"Testing {model_name} with {model_path}")
            
            req_data = self.vllm_model_example_map[model_name](questions=[question], modality=modality, model_path=model_path)
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
            
            inputs = {
                "prompt": req_data.prompts[0],
                "multi_modal_data": {
                    modality: image
                },
            }   
            
            ans = None
            for _ in range(3):
                try:
                    outputs = llm.generate(inputs, sampling_params=sampling_params)
                    ans = outputs[0].outputs[0].text
                    break
                except Exception as e:
                    print (f'Test {model_name} failed.')
                    print (e)
            
            ans_dict[model_name] = ans
            
        json.dump(ans_dict, open(out_json_path, 'w'), indent=4)
            
        
if __name__ == '__main__':
    image_path = './data/test_imgs/img.jpg'
    prompt = 'Describe the image.'
    
    test_vllm_models = TestVLLMModels()
    test_vllm_models.test_vllm_models(image_path, prompt)