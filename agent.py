from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        with torch.no_grad():
            return self.pipe(message, **kwargs)