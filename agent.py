from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
import boto3
import json
from .utils.prompt import convert_format, convert_non_system_prompts
class BedrockLLMs:
    def __init__(self,model_id):
        self.model_id = model_id

class CoreLLMs:
    def __init__(self,
                 model_name = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 quantization='auto',
                 ) -> None:
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.quantization = quantization
        self.is_agent_initialized = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.host = 'local'
        self._initialize_agent()
        
    def _initialize_agent(self):
        
        quantize_config = None
        
        if self.quantization == 'int4':
            print("Using int4")
            quantize_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            
        elif self.quantization == 'int8':
            print("Using int8")
            quantize_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map = self.device, 
            torch_dtype = 'auto', 
            trust_remote_code = True,
            quantization_config = quantize_config 
        )
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        self.pipe = pipeline("text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
        )
        
    def _delete_agent(self):
        if self.is_agent_initialized:
            del self.pipe
            del self.model
            gc.collect()
            torch.cuda.empty_cache() 
            self.is_agent_initialized = False
            print("Agent deleted")

    def __call__(self, message, **kwargs):
        if not self.is_agent_initialized:
            self._initialize_agent()
            self.is_agent_initialized = True
        if 'llama' not in self.model_name.lower():
            message = convert_non_system_prompts(message)
        with torch.no_grad():
            return self.pipe(message, **kwargs)[0]['generated_text']
        
        
class BedRockLLMs:
    def __init__(self,
                model_name = "meta.llama3-8b-instruct-v1:0", 
                access_key = None,
                secret_key = None,
                 ) -> None:
        self.client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        self.model_id = model_name
        self.host = 'cloud'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _initialize_agent(self):
        pass
    
    def _delete_agent(self):
        pass
    
    def __call__(self, message, **kwargs):
        prompt = convert_format(message)
        request = {
            "prompt": prompt,
            # Optional inference parameters:
            "max_gen_len": 1536,
            "temperature": 0.4,
            "top_p": 0.9,
        }
        response = self.client.invoke_model(body=json.dumps(request), modelId=self.model_id, contentType='application/json')
        return json.loads(response['body'].read().decode('utf-8'))['generation']
    
if __name__ == "__main__":
    llm  = BedRockLLMs()
    print(llm([{'role':'system','content':'You are a friendly assistant'},{'role':'user','content':'How are you today'}]))