import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class OpenRouter:
    def __init__(self, system_prompt='You are a helpful assistant.'):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
    def infer(self, model_name='meta-llama/llama-4-scout', image_path=None, prompt=None):
        '''
        support model list:
        meta-llama/llama-4-scout: https://openrouter.ai/meta-llama/llama-4-scout/api
        meta-llama/llama-4-maverick: https://openrouter.ai/meta-llama/llama-4-maverick/api
        
        google/gemma-3-4b-it: https://openrouter.ai/google/gemma-3-4b-it/api
        google/gemma-3-12b-it: https://openrouter.ai/google/gemma-3-12b-it/api
        google/gemma-3-27b-it: https://openrouter.ai/google/gemma-3-27b-it/api
        
        x-ai/grok-2-vision-1212: https://openrouter.ai/x-ai/grok-2-vision-1212/api
        
        openai/gpt-4o: https://openrouter.ai/openai/gpt-4o/api
        openai/gpt-4o-mini: https://openrouter.ai/openai/gpt-4o-mini/api
        
        google/gemini-2.5-pro-preview-03-25: https://openrouter.ai/google/gemini-2.5-pro-preview-03-25/api
        google/gemini-2.5-flash-preview: https://openrouter.ai/google/gemini-2.5-flash-preview/api
        
        anthropic/claude-3.5-sonnet: https://openrouter.ai/anthropic/claude-3.5-sonnet/api
        '''
        base64_image = encode_image(image_path=image_path)
        
        completion = self.client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model=model_name,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ]
        )
        
        ans = completion.choices[0].message.content
        
        return ans
    
    
    def infer_text(self, model_name='openai/gpt-4o-mini', prompt=None):
        '''
        support models: openai/gpt-4o-mini
        '''
        
        completion = self.client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model=model_name,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
                }
            ]
        )
        
        ans = completion.choices[0].message.content
        
        return ans