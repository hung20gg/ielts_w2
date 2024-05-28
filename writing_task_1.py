from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import gc
import torch

from .utils.extract import clean_output
from .utils.prompt import get_system_prompt_wt1, get_instruction_prompt_wt1


from .writing import AgentWT2

class VisionModel:
    def __init__(self, model_id, quantization='auto', initialize = False, allow_delete = True):
        self.model_id = model_id
        self.quantization = quantization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.processor = None
        self.is_agent_initialized = False
        self.allow_delete = allow_delete
        
        self.generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.3, 
            "do_sample": False, 
        }
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        if initialize:
            self._initialize_vision_model()
    
    def _initialize_vision_model(self):   
        quantize_config = None
        if self.quantization == 'int4':
            print("Using int4")
            quantize_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, 
                                                            device_map=self.device, 
                                                            trust_remote_code=True, 
                                                            torch_dtype="auto", 
                                                            quantization_config=quantize_config)
        
        self.is_agent_initialized = True

    def _delete_vision_model(self):
        if self.is_agent_initialized and self.allow_delete:
            del self.processor
            del self.model
            gc.collect()
            torch.cuda.empty_cache() 
            self.is_agent_initialized = False
            print("Vision model deleted")
            
    def __call__(self, description, image, is_chart = True):
        if not self.is_agent_initialized:
            self._initialize_vision_model()
            
        if isinstance(image, str):
            image = Image.open(image)
        
        messages = [ 
            {"role": "user", "content": f"<|image_1|>\n{description}. Can you provide insightful, detail and precise information about the {'chart' if is_chart else 'diagram'} (you can ignore the color)?"}, 
        ] 
        with torch.no_grad():
            prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device) 
            
            generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **self.generation_args) 
            
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            
            return response.split('!')[-1].strip()
            

class AgentWT1(AgentWT2):
    def __init__(self, pipe, 
                 role = 'assistant', # Chatbot role
                 agent = 'task1',
                 vision_model = "microsoft/Phi-3-vision-128k-instruct",
                 model_embedding = 'sentence-transformers/all-mpnet-base-v2',
                 explain_metric = False, 
                 verbose = False,
                 rag = False,
                 vision_quantization = 'auto', 
                 initialize = False, 
                 allow_delete = True,
                 **generation_args,
                 ) -> None:
        
        super().__init__(pipe, role, agent, model_embedding, explain_metric, verbose, rag, **generation_args)
        
        self.vision_model_name = vision_model
        self.agent = 'task1'

        self.vision_model = VisionModel(self.vision_model_name, quantization=vision_quantization, initialize=initialize, allow_delete=allow_delete)
        self.allow_delete = allow_delete
        
        self.system_prompt = self.get_system_prompt()
        
        
        if generation_args:
            self.generation_args = generation_args
        else:
            self.generation_args = {
                "max_new_tokens": 2000,
                "return_full_text": False,
                "temperature": 0.6,
                "top_p":0.9,
                "do_sample": True,  
            }
    # Rewrite some method
    def get_system_prompt(self, explain_metric=False):
        return get_system_prompt_wt1(explain_metric)
    
    def get_instruction_prompt(self):
        return get_instruction_prompt_wt1(self.topic, self.essay, self.verbal_description)
        
    
    def _delete_vision_model(self):
        if self.llm.host == 'local':
            self.vision_model._delete_vision_model()
                
        
    def describe_image(self, chart = True):
        if self.allow_delete:
            self.llm._delete_agent()
        return self.vision_model(self.description, self.image_path, chart)
    
    def _process_image(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path)
            # image = np.array(image)
        return image
    
    def show_image(self):
        self.image.show()
    
    def generate_response(self, image_path: str, essay_topic: str, student_response: str, is_rescore = True, is_chart = None):
        
        if self.generated:
            print("Already generated, clearing messages if you want to generate again")
            return self.messages[-1]['content']
        
        self.image = self._process_image(image_path)
        self.image_path = image_path
        self.rescore = is_rescore
        self.essay = student_response
        self.topic = essay_topic
        self.description = essay_topic.split('\n')[0]
        
        if not isinstance(is_chart,bool):
            chart_text = ['chart', 'table']
            is_chart = False
            for text in chart_text:
                if text in essay_topic:
                    is_chart = True
                    break
                
        
        verbal_description = self.describe_image(is_chart)
        
        self.verbal_description = verbal_description
        self._delete_vision_model()
        # Some prompt here
        
        self.instruction_prompt = self.get_instruction_prompt()
        self.messages.append({"role": "user", "content": self.instruction_prompt})
        # if not self.is_rag:
        #     self.messages.append({"role": "user", "content": self.instruction_prompt})
        # else:
        #     self.incontext_prompt(essay_topic, student_response)
        
        self.generated = True
        output = self.llm(self.messages)
        self.messages.append({"role": self.role, "content": output},)
        # return output[0]['generated_text']
        
        adjusted_output = self._until_correct()
        self.original_output = clean_output(adjusted_output)
        self.rag_messages = self.messages.copy()
        self.messages = self.system_messages.copy()
        
        self.messages.append({"role": "user", "content": self.instruction_prompt})
        self.messages.append({"role": self.role, "content": self.original_output})
        
        if self.rescore:
            self._rescore()
            rescore_output = self._until_correct()
            self.rescore_output = clean_output(rescore_output)
            return self.rescore_output
        return self.original_output