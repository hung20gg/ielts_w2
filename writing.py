from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .utils.extract import get_score, clean_output, get_essay
from .utils.prompt import get_system_prompt, get_instruction_prompt, get_incontext_prompt,  get_output_suggestion_format
import torch

from .utils.embedding import ClusterRAG

class AgentWT2:
    
    # This class will only use prompting technique to generate the scoring
    
    def __init__(self, 
                 llm,
                 role='assistant', 
                 agent = 'task2',
                 model_embedding = 'sentence-transformers/all-mpnet-base-v2',
                 explain_metric = True, 
                 verbose = False,
                 rag = False, **generation_args):
        
        self.explain_metric = explain_metric
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.system_prompt = self.get_system_prompt(explain_metric)
        self.llm = llm
        self.model_name = self.llm.model_name
        self.model_embedding = model_embedding
        self.agent = agent
        self.output_suggestion = get_output_suggestion_format()
        self.essay_type = 'essay'
        
        self.role = role
        self._setup_system_messages()
        
        if generation_args:
            self.generation_args = generation_args
        else:
            self.generation_args = {
                "max_new_tokens": 2000,
                "return_full_text": False,
                "temperature": 0.4,
                "top_p":0.9,
                "do_sample": True,  
            }
        self.mode = 'harsh'
        
        self.is_rag = rag
        if rag:
            self.setup_rag(model_embedding)
            
        self.verbose = verbose
        self.clear_messages()
    
    def _setup_system_messages(self):
        if 'llama' in self.model_name:
            self.mode = 'harsh'
        self.system_messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        
    def setup_rag(self, model_embedding='sentence-transformers/all-mpnet-base-v2', df_path='data/lookup_essay.csv'):
        self.is_rag = True
        self.rag = ClusterRAG(model_embedding, is_cluster=False, df_path=df_path)
        self.clear_messages()
    
    def remove_rag(self):
        self.is_rag = False
        self.rag = None
        self.clear_messages()
    
    def clear_messages(self):
        self._setup_system_messages()
        self.messages = self.system_messages.copy()
        self.generated = False
        self.rescore = False
        self.general = 0
        self.tr = 0
        self.cc = 0
        self.lr = 0
        self.gr = 0
        
        self.topic = ''
        self.essay = ''
        self.revived_essay = ''
        self.original_output = ''
        self.rescore_output = ''
        self.feed_back = ''
        self.instruction_prompt = ''
    
    def get_system_prompt(self, explain_metric = False):
        return get_system_prompt(explain_metric)
        
        
    def add_message(self, message):
        self.messages.append(message)
        
    def get_message(self):
        return self.messages
    
    def get_topic(self):
        return self.topic
    
    def get_essay(self):
        return self.essay
    
    def get_revived_essay(self):
        return self.revived_essay
    
    def get_original_output(self):
        return self.original_output
    
    def get_rescore_output(self):
        return self.rescore_output
    
    def get_score(self, verbose=False):
        if verbose:
            print(f"General: {self.general}, \nTask Response: {self.tr}, \nCoherence and Cohesion: {self.cc}, \nLexical Resource: {self.lr}, \nGrammatical Range and Accuracy: {self.gr}")
        return self.general, self.tr, self.cc, self.lr, self.gr
    
    def change_system_prompt(self, new_prompt = None):
        if new_prompt:
            self.system_prompt = new_prompt
            return
        print("Please provide new prompt, if not the prompt remain the same")        
    
    def change_system_prompt_criteria(self, criteria = False):
        self.explain_metric = criteria
        self.system_prompt = self.get_system_prompt(criteria)
        self.clear_messages()
        
    
    def _check_score_valid(self, check_int=False):
        if (self.tr + self.cc + self.lr + self.gr)//2 != self.general * 2:
            return False
        
        if check_int:
            if int(self.tr) != self.tr or int(self.cc) != self.cc or int(self.lr) != self.lr or int(self.gr) != self.gr:
                return False
        return True
    
    
    def _incorrect_data(self):
        if not self._check_score_valid():
            if self.verbose:
                print("Inconsistent scoring")
            
            proposed_correct_score = ((self.tr + self.cc + self.lr + self.gr)//2)/2
            add_prompt = {"role": "user", "content": f"Your score doesn't make sense. How can I get a {self.general} in general while I only get {self.tr} in Task Response, {self.cc} in Coherence and Cohesion, {self.lr} in Lexical Resource and {self.gr} in Grammatical Range and Accuracy. You need to check the grade of each criteria and overall score again. Maintain the output format"},
            self.messages.append(add_prompt[0])
            output = self.llm(self.messages, **self.generation_args)
            self.messages.append({"role": self.role, "content": output},)
            
            return output
        else:
            return self.messages[-1]['content']
    
    
    def _incorrect_data2(self):
        if not self._check_score_valid():
            if self.verbose:
                print("Inconsistent scoring")
                
            proposed_correct_score = ((self.tr + self.cc + self.lr + self.gr)//2)/2
            
            add_prompt = {"role": "user", "content": f"Your score is not consistence and still does not sum up equally. You score this {self.essay_type} {self.tr} in Task Response, {self.cc} in Coherence and Cohesion, {self.lr} in Lexical Resource and {self.gr} in Grammatical Range and Accuracy, so the total score should be {proposed_correct_score}. You should grade the {self.essay_type} again, and maintain the output format"},
            self.messages.append(add_prompt[0])
            output = self.llm(self.messages, **self.generation_args)
            self.messages.append({"role": self.role, "content": output},)
            
            return output
        else:
            return self.messages[-1]['content']
        
    def _reprompt(self):

        add_prompt = {"role": "user", "content": f"Your score seem to have some mistake in term of logic. You should reevaluate your score and remain the output format."},
        self.messages.append(add_prompt[0])
        output = self.llm(self.messages, **self.generation_args)
        self.messages.append({"role": self.role, "content": output},)
        return output
    
    def _rescore(self):
        adj = 'strict' if self.mode == 'harsh' else 'loose'
        
        self.messages.append({"role": "user", "content": f"Do you think you are too {self.mode}? Your explanation for each metric seems to be too {adj} and may not fit with its criteria. You should reevaluate your score and remain the output format and make sure all the criteria scores is integer and only general score can be float (round to .5)."})
        
        output = self.llm(self.messages, **self.generation_args)
        self.messages.append({"role": self.role, "content": output},)
        return output
    
    
    def _not_integer(self):
    
        flag = False
        prompt_tr_score = ''
        prompt_cc_score = ''
        prompt_lr_score = ''
        prompt_gr_score = ''
        
        if int(self.tr) != self.tr:
            flag = True
            prompt_tr_score = f'Task Response score is {self.tr}, which is not an integer. '
        
        if int(self.cc) != self.cc:
            flag = True
            prompt_cc_score = f'Coherence and Cohesion score is {self.cc}, which is not an integer. '
            
        if int(self.lr) != self.lr:
            flag = True
            prompt_lr_score = f'Lexical Resource score is {self.lr}, which is not an integer. '
            
        if int(self.gr) != self.gr:
            flag = True
            prompt_gr_score = f'Grammatical Range an Accuracy score is {self.gr}, which is not an integer. '    
            
        if flag == True:
            if self.verbose:
                print("Not integer")
            
            add_prompt = {"role": "user", "content": f"Your score seem incorrect. {prompt_tr_score}{prompt_cc_score }{prompt_lr_score }{prompt_gr_score}. You should look again your score and make sure all the criteria scores is integer and only general score can be float (round to .5). Maintain the output format"},
            self.messages.append(add_prompt[0])
            output = self.llm(self.messages, **self.generation_args)
            self.messages.append({"role": self.role, "content": output},)
            return output
        else:
            return self.messages[-1]['content']
        
        
    def _until_correct(self):
        """
        This function will keep adjusting the score until it is correct
        
        Adjusting score will be done by reprompting until the score satisfied or reach a certain limit
        """
        loop_count = 0
        self.general, self.tr, self.cc, self.lr, self.gr = get_score(self.messages[-1]['content'])
        result = self.messages[-1]['content']
        while not self._check_score_valid(check_int=True):
            loop_count+=1
            
            if self.verbose:
                print(f"______Adjusting score {loop_count}_______")
                
            if loop_count%3 == 0:
                result = self._reprompt()
            elif loop_count%2 == 0:
                result = self._incorrect_data2()
            else:
                result = self._incorrect_data()
            
            self.general, self.tr, self.cc, self.lr, self.gr = get_score(result)
            result = self._not_integer()
            self.general, self.tr, self.cc, self.lr, self.gr = get_score(result)
            if loop_count >= 4:
                break
        
        return result
    
    def incontext_prompt(self, essay_topic, student_response):
        """ Using RAG to get incontext example
        """
        system_prompt_1, system_prompt_2 = self.get_system_prompt(incontext=True, explain=self.explain_metric)
        
        if 'llama' in self.model_name:
            self.messages = [
                {"role": "system", "content": system_prompt_1},
            ]
        else:
            self.messages = [
                {"role": "user", "content": system_prompt_1},
                {"role": self.role, "content": "Sure, I will be happy to help you with that"},
            ]
        
        topics, essays, comments = self.rag.retrieve(essay_topic, student_response, topk=3)
        for topic, essay, comment in zip(topics, essays, comments):
            incontext_prompt = get_incontext_prompt(topic, essay)
            self.messages.append({"role": "user", "content": incontext_prompt})
            self.messages.append({"role": self.role, "content": comment})

        self.messages.append({"role": "user", "content": system_prompt_2 + get_instruction_prompt(self.topic, self.essay)})
    
    def get_instruction_prompt(self):
        return get_instruction_prompt(self.topic, self.essay)
    
    
    def generate_response(self, essay_topic, student_response, is_rescore=True):
        """Receive the essay topic and student response, then generate the scoring for the essay

        Args:
            essay_topic (str): the topic of the essay
            student_response (str): the student response
            is_rescore (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: chatbot response
        """
        if self.generated:
            print("Already generated, clearing messages if you want to generate again")
            return self.messages[-1]['content']
        
        self.rescore = is_rescore
        self.essay = student_response
        self.topic = essay_topic
        
        self.instruction_prompt = self.get_instruction_prompt()
        
        if not self.is_rag:
            self.messages.append({"role": "user", "content": self.instruction_prompt})
        else:
            self.incontext_prompt(essay_topic, student_response)
        
        self.generated = True
        output = self.llm(self.messages, **self.generation_args)
        self.messages.append({"role": self.role, "content": output},)
        # return output
        
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
    
    
    def indepth_feedback(self, force_original=False):
        if not self.generated:
            print("You need to generate response first")
            return
        if force_original:
            first_output = self.original_output
        else:
            first_output = self.messages[-1]['content']
        
        self.messages = self.system_messages.copy()
        self.messages.append({"role": "user", "content": self.instruction_prompt})
        self.messages.append({"role": self.role, "content": first_output})
        self.messages.append({"role": "user", "content": f"Provide more in-depth feedback in the {self.essay_type}, list out all the mistake made in the {self.essay_type} and suggest how to improve it. It is good to list all the grammatical mistake such as spelling, punctuation, grammar. \n {self.output_suggestion}"})
        
        output = self.llm(self.messages, **self.generation_args)
        self.messages.append({"role": self.role, "content": output},)
        
        if not force_original:
            self.feed_back = output    
        
        return self.feed_back 
    
    def revive_essay(self, MAX_TARGET = 7.5, ADJUSTMENT_MODE = 'close_to_original'):
        if not self.generated:
            print("You need to generate response first")
            return
        
        if self.rescore and ADJUSTMENT_MODE == 'close_to_original':
            self.indepth_feedback(force_original=True)
        
        if self.general <= MAX_TARGET - 0.5:
            self.messages.append({"role": "user", "content": f"Making adjustment directly into the {self.essay_type} to improve at least 1 score in each metric. {self.output_suggestion}"})
            suggest_essay = self.llm(self.messages, **self.generation_args)
            
        else: 
            self.messages.append({"role": "user", "content": f"Making adjustment directly into the {self.essay_type} to optimize it. {self.output_suggestion}"})
            suggest_essay = self.llm(self.messages, **self.generation_args)
            
        self.messages.append({"role": self.role, "content": suggest_essay},)
        
        self.revived_essay = get_essay(suggest_essay)
        return suggest_essay
    