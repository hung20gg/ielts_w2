from PIL import Image
import numpy as np
from transformers import pipeline
import gc
import torch

from ielts_scoring.utils.extract import clean_output
from ielts_scoring.utils.prompt import get_system_prompt_wt1, get_instruction_prompt_wt1

from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model


from ielts_scoring.writing import AgentWT2

class AgentWT1(AgentWT2):
    def __init__(self, pipe, 
                 role = 'assistant', # Chatbot role
                 agent = 'task1',
                 diagram_model = "xtuner/llava-phi-3-mini-hf",
                 chart_model = "mPLUG/TinyChart-3B-768",
                 model_embedding = 'sentence-transformers/all-mpnet-base-v2',
                 explain_metric = False, 
                 verbose = False,
                 rag = False, 
                 **generation_args,
                 ) -> None:
        
        super().__init__(pipe, role, agent, model_embedding, explain_metric, verbose, rag, **generation_args)
        
        self.diagram_model_name = diagram_model
        self.chart_model_name = chart_model
        self.agent = 'task1'
        
        self.is_diagram = False
        self.is_chart = False

        self.diagram_pipe = None
        
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
        
    def _initialize_diagram(self):
        
        self.is_diagram = True
        self._delete_chart()
        self.llm._delete_agent()
        
        self.diagram_pipe = pipeline("image-to-text",
            model = self.diagram_model_name,
            torch_dtype=torch.bfloat16, device=0
        )
        
    def _initialize_chart(self):
        
        self.is_chart = True
        self._delete_diagram()
        self.llm._delete_agent()
        
        self.chart_tokenizer, self.chart_model, self.chart_image_processor, self.chart_context_len = load_pretrained_model(
            self.chart_model_name, 
            model_base=None,
            model_name=get_model_name_from_path(self.chart_model_name),
            device="cuda"
        )
        
    def _delete_diagram(self):
        if self.is_diagram:
            del self.diagram_pipe
            self.is_diagram = False
            
            gc.collect()
            torch.cuda.empty_cache() 
            print("Diagram deleted")
            
    def _delete_chart(self):
        if self.is_chart:
            del self.chart_model
            del self.chart_tokenizer
            del self.chart_image_processor
            del self.chart_context_len
            self.is_chart = False
            
            gc.collect()
            torch.cuda.empty_cache() 
            print("Chart deleted")
            
    def _delete_agent(self):
        self._delete_diagram()
        self._delete_chart()
        print("Agent deleted")
        
    def describe_chart(self):
        with torch.no_grad():
            verbal_description = inference_model([self.image_path], self.description + '. Describe it.', 
                                                 self.chart_model, self.chart_tokenizer, self.chart_image_processor, self.chart_context_len, conv_mode="phi", max_new_tokens=2048)
            return verbal_description
    def describe_diagram(self):
        with torch.no_grad():
            prompt = f"<|user|>\n<image>\n{str(self.description)}. Extract the features in it.\n<|assistant|>\n"
            outputs = self.diagram_pipe(self.image, prompt=prompt, generate_kwargs={"max_new_tokens": 1000})
            return outputs[0]['generated_text']
    
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
                
        if is_chart:
            self._initialize_chart()
            verbal_description = self.describe_chart()
        else:
            self._initialize_diagram()
            verbal_description = self.describe_diagram()
        
        self.verbal_description = verbal_description
        self._delete_agent()
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