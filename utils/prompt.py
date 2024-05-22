
from .extract import get_score
import os
# print()
DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_output_suggestion_format():
    with open(DIR+'/prompt/output_suggestion_format.txt', 'r') as f:
        output_suggestion_format = f.read()
    return output_suggestion_format


def get_system_prompt_wt1(explain = False):
    with open(DIR+'/prompt/explain_metric_short.txt', 'r') as f:
        explain_metric = f.read()
        
    with open(DIR+'/prompt/output_format.txt', 'r') as f:
        output_format = f.read()
        
    system_prompt_wt1 =f"""
You are an English teaching assistant, and you are good at evaluating essays and reports, and your students need you for their IELTS Academic writing task 1. You will make evaluations based on the topic and the student's response. 
However, since you cannot see the images of charts or diagrams, you will be given a verbal description of them. Sometimes, the description may not be very detailed, or they might contain false information, so just use it as a reference.

You should grade the report's general score and its 4 metrics in IELTS Writing task 1, which are Task Response, Coherence and Cohesion, Lexical Resource and Grammatical Range and Accuracy. 
The overall must be the mean value of 4 metrics' score and can be a float value between 1 and 9 (round to .5), but each metric score must be an integer between 0 and 9.
Recall the IELTS Writing task 1 band score criteria.
{explain_metric if explain else ''}
The formula to calculate the general score is:
```
General_score = ( ( Task_Response + Coherence_and_Cohesion + Lexical_Resource + Grammatical_Range_and_Accuracy) // 2 ) / 2 
```

In each metric, you should give a detailed explanation and point out exactly the student mistakes that led to that score.
Your output format should be like this:
{output_format}

Here is some tips for you:
- Most of the reports should be written in the third person.
- Be easy on grading report.
- The report should not have any subjective and personal opinion.
- The report's grade range is mostly around 5.5 to 8.0.
- Every report that has more than 150 words with little spelling or grammatical errors will score at least a 5.5.
- Any attempt of using complex sentences might have at least 4.5 in Grammatical Range and Accuracy.
- The overall score should be the mean value of 4 metric scores, round down to .0 and .5, so make sure your evaluation right.
- Provide constructive feedback.

"""  
    return system_prompt_wt1
    

def get_system_prompt(explain = False, incontext = False):
    
    # with open(DIR+'/prompt/explain_metric_short.txt', 'r') as f:
    #     explain_metric = f.read()
    
    explain_metric =''
        
    with open(DIR+'/prompt/output_format.txt', 'r') as f:
        output_format = f.read()
    
    if incontext:
        system_prompt_1 = "You are an English teaching assistant, and you are good at evaluating essays, and your students need you for their IELTS Academic essay task 2. You will make evaluation based on the topic and student's response."
        system_prompt_2 = f"""
You should grade the essay's general score and its 4 metrics in IELTS Writing task 2, which are Task Response, Coherence and Cohesion, Lexical Resource and Grammatical Range and Accuracy. 
The overall must be the mean value of 4 metrics' score and can be a float value between 1 and 9 (round to .5), but each metric's score must be an integer between 0 and 9.
Recall the IELTS Writing task 2 band score criteria.
{explain_metric if explain else ''}
The formula to calculate the general score is:
General_score = ( ( Task_Response + Coherence_and_Cohesion + Lexical_Resource + Grammatical_Range_and_Accuracy) // 2 ) / 2 
```

In each metric, you should give a detailed explanation and point out exactly the student mistakes that led to that score.
Your output format should be like this:
{output_format}

Here is some tips for you:
- Most of the essay should be written in the third person.
- Be easy on grading essay.
- The essay's grade range is mostly around 5.5 - 7.5.
- Every essay that has more than 250 words with little spelling or grammatical errors will score at least a 5.0.
- Any attempt of using complex sentences might have at least 4.5 in Grammatical Range and Accuracy.
- The overall score should be the mean value of 4 metric scores, round down to .0 and .5, so make sure your evaluation right.
- Provide constructive feedback.
"""
    
        return system_prompt_1, system_prompt_2
    
    
    system_prompt = f"""
You are an English teaching assistant, and you are good at evaluating essays, and your students need you for their IELTS Academic essay task 2. You will make evaluations based on the topic and the student's response.
You should grade the essay's general score and its 4 metrics in IELTS Writing task 2, which are Task Response, Coherence and Cohesion, Lexical Resource and Grammatical Range and Accuracy. 
The overall must be the mean value of 4 metrics' score and can be a float value between 1 and 9 (round to .5), but each metric's score must be an integer between 0 and 9.
Recall the IELTS Writing task 2 band score criteria.
{explain_metric if explain else ''}
The formula to calculate the general score is:
```
General_score = ( ( Task_Response + Coherence_and_Cohesion + Lexical_Resource + Grammatical_Range_and_Accuracy) // 2 ) / 2 
```

In each metric, you should give a detailed explanation and point out exactly the student mistakes that led to that score.
Your output format should be like this:
{output_format}

Here is some tips for you:
- Most of the essay should be written in the third person.
- Be easy on grading essay.
- The essay's grade range is mostly around 5.5 to 7.5.
- Every essay that has more than 250 words with little spelling or grammatical errors will score at least a 5.0.
- Any attempt of using complex sentences might have at least 4.5 in Grammatical Range and Accuracy.
- The overall score should be the mean value of 4 metric scores, round down to .0 and .5, so make sure your evaluation right.
- Provide constructive feedback.
"""
    
    return system_prompt


def get_instruction_prompt(essay_topic, student_response):
    instruction_prompt = f"""
You will have to grade this essay in IELTS WRITING task 2 academic guideline
Topic of the essay: {essay_topic}
Essay: 
{student_response}
Make sure that you follow the IELTS WRITING task 2 guideline to grade this essay, and do the correct evaluation of general score.
Let's evaluate step by step.
    """
    return instruction_prompt

def get_instruction_prompt_wt1(essay_topic, student_response, verbal_description):
    instruction_prompt_wt1 = f"""
You will have to grade this report in IELTS WRITING task 1 academic guideline
Topic: 
{essay_topic}
Description of the chart or diagram:
{verbal_description}
Student's Report: 
{student_response}
Make sure that you follow the guideline to grade this essay, and do the correct evaluation of general score.
Let's evaluate step by step.
"""
    return instruction_prompt_wt1

def get_incontext_prompt(topic, example_essay):
    incontext_prompt = f"""
You will have to give comment to this IELTS WRITING task 2 essay.
Topic: 
{topic}
Student's Essay:
{example_essay}
"""
    return incontext_prompt


def incorrect_data(pipe, messages, general, tr, cc, lr, gr, verbose=False, **generation_args):
    if (tr + cc + lr + gr)//2 != general * 2:
        
        if verbose:
            print("Inconsistent scoring")
        
        proposed_correct_score = ((tr + cc + lr + gr)//2)/2
        add_prompt = {"role": "user", "content": f"Your score doesn't make sense. How can I get a {general} in general while I only get {tr} in Task Response, {cc} in Coherence and Cohesion, {lr} in Lexical Resource and {gr} in Grammatical Range and Accuracy, which the general result is {proposed_correct_score}? You need to check the grade again, and maintain the output format"},
        messages.append(add_prompt[0])
        output = pipe(messages, **generation_args["model_args"])
        messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
        return output[0]['generated_text']
    else:
        return messages[-1]['content']


def incorrect_data2(pipe, messages, general, tr, cc, lr, gr, verbose=False, **generation_args):
    if (tr + cc + lr + gr)//2 != general * 2:
        
        if verbose:
            print("Inconsistent scoring")
        
        proposed_correct_score = ((tr + cc + lr + gr)//2)/2
        add_prompt = {"role": "user", "content": f"Your score is not consistence and still does not sum up equally. You score this essay {tr} in Task Response, {cc} in Coherence and Cohesion, {lr} in Lexical Resource and {gr} in Grammatical Range and Accuracy, so the general score should be {proposed_correct_score}, which is different than your original score of {gr}. You should grade the essay again, and maintain the output format"},
        messages.append(add_prompt[0])
        output = pipe(messages, **generation_args["model_args"])
        messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
        return output[0]['generated_text']
    else:
        return messages[-1]['content']
    
    
def reprompt(pipe, messages, **generation_args):

    add_prompt = {"role": "user", "content": f"Your score seem to have some mistake in term of logic. You should reevaluate your score and remain the output format."},
    messages.append(add_prompt[0])
    output = pipe(messages, **generation_args["model_args"])
    messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
    return output[0]['generated_text']



def rescore(pipe, messages, mode, **generation_args):
    adj = 'strict' if mode == 'harsh' else 'loose'
    messages.append({"role": "user", "content": f"Do you think you are too {mode}? Your explanation for each metric seems to be too {adj} and may not fit with its criteria. You should reevaluate your score and remain the output format. Make sure all the criteria scores is integer and only general score can be float (round to .5)."})
    
    output = pipe(messages, **generation_args["model_args"])
    messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
    return output[0]['generated_text']


def indepth_feedback(pipe, messages, **generation_args):
    messages.append({"role": "user", "content": f"Provide more detailed feedback in the essay, including all the mistake made in the essay and how to improve it."})
    output = pipe(messages, **generation_args["model_args"])
    messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
    return output[0]['generated_text']


def until_correct(pipe, messages, verbose=False, **generation_args):
    loop_count = 0
    general, tr, cc, lr, gr = get_score(messages[-1]['content'])
    result = messages[-1]['content']
    while (tr + cc + lr + gr)//2 != general*2 or int(tr) != tr or int(cc) != cc or int(lr) != lr or int(gr) != gr:
        loop_count+=1

        if verbose:
            print(f"______Adjusting score {loop_count}_______")
        
        if loop_count%3 == 0:
            result = reprompt(pipe, messages**generation_args)
        
        if loop_count%2 == 0:
            result = incorrect_data2(pipe, messages, general, tr, cc, lr, gr, verbose, **generation_args)
        else:
            result = incorrect_data(pipe, messages, general, tr, cc, lr, gr, verbose, **generation_args)
        
        general, tr, cc, lr, gr = get_score(result)
        result = not_integer(pipe, messages, tr, cc, lr, gr, **generation_args)
        general, tr, cc, lr, gr = get_score(result)
        
        if loop_count > 4:
            break
    
    return result


def not_integer(pipe, messages, tr, cc, lr, gr, **generation_args):
    
    flag = False
    prompt_tr_score = ''
    prompt_cc_score = ''
    prompt_lr_score = ''
    prompt_gr_score = ''
    
    if int(tr) != tr:
        flag = True
        prompt_tr_score = f'Task Response score is {tr}, which is not an integer. '
    
    if int(cc) != cc:
        flag = True
        prompt_cc_score = f'Coherence and Cohesion score is {cc}, which is not an integer. '
        
    if int(lr) != lr:
        flag = True
        prompt_lr_score = f'Lexical Resource score is {lr}, which is not an integer. '
        
    if int(gr) != gr:
        flag = True
        prompt_gr_score = f'Grammatical Range an Accuracy score is {gr}, which is not an integer. '    
        
    if flag == True:
        print("Not integer")
        
        add_prompt = {"role": "user", "content": f"Your score seem incorrect. {prompt_tr_score}{prompt_cc_score }{prompt_lr_score }{prompt_gr_score}. You should look again your score and make sure all the criteria scores is integer and only general score can be float (round to .5). Maintain the output format"},
        messages.append(add_prompt[0])
        output = pipe(messages, **generation_args["model_args"])
        messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
        return output[0]['generated_text']
    else:
        return messages[-1]['content']
    

def convert_format(data, has_system=True):
    prompt = ""
    sys_non_llama = ''
    for i, item in enumerate(data):
        if i == 0 and item["role"] == "system":
            if has_system:
                system_message = item["content"]
                prompt += f"<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n{system_message}<|eot_id|>\n"
            else:
                sys_non_llama = item["content"]+'\n'

        elif item["role"] == "user":
            if i==1 and not has_system:
                user_message = item["content"]
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{sys_non_llama}{user_message}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
            else:
                user_message = item["content"]
                prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_message}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        elif item["role"] == "assistant":
            assistant_message = item["content"]
            prompt += f"{assistant_message}\n<|eot_id|>"

    return prompt.rstrip("\n")

def convert_non_system_prompts(messages):
    new_messages = []
    if messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        for i in range(1,len(messages)):
            message = messages[i]['content']
            if i == 1:
                message = system_prompt + '\n' + message
            new_messages.append({"role": messages[i]['role'], "content": message})
        return new_messages
    return messages

# if __name__ == "__main__":
    # print(get_system_prompt_wt1())
    # print(get_system_prompt())
    # print(get_instruction_prompt("This is a topic", "This is a response"))
    # print(get_instruction_prompt_wt1("This is a topic", "This is a response", "This is a description"))
    # print(get_incontext_prompt("This is a topic", "This is an example essay"))
    # print(get_output_suggestion_format())
    # print(convert_format([{'role':'system','content':'You are a friendly assistant'},{'role':'user','content':'How are you today'}]))
    # print(convert_non_system_prompts([{'role':'system','content':'You are a friendly assistant'},{'role':'user','content':'How are you today'}]))