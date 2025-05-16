import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_API_MAP
from apis.open_router import OpenRouter


class TestApis:
    def __init__(self):
        self.model_api_map = MODEL_API_MAP
        
        self.open_router = OpenRouter()
        self.qwen_vl = QwenVL()
        
    def test_open_router(self, image_path, prompt):
        out_json_path = './data/tests/open_router.json'
        ans_dict = {}
        
        for model_name, api_name in self.model_api_map['open_router'].items():
            print (f"Testing {model_name} with {api_name}")
            
            ans = None
            for _ in range(3):
                try:
                    ans = self.open_router.infer(api_name, image_path, prompt)
                    ans_dict[model_name] = ans
                    print (f'Test {model_name} done.')
                    break
                except Exception as e:
                    print (f'Test {model_name} failed.')
                    print (e)
                    continue
            
        json.dump(ans_dict, open(out_json_path, 'w'), indent=4)


if __name__ == '__main__':
    test_apis = TestApis()
    
    image_path = './data/test_imgs/img.jpg'
    prompt = 'Describe the image.'
    
    test_apis.test_open_router(image_path, prompt)