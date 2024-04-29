
from .extract import get_score

def get_output_suggestion_format():
    with open('prompt/output_suggestion_format.txt', 'r') as f:
        output_suggestion_format = f.read()
    return output_suggestion_format

def get_system_prompt():
    
    with open('prompt/explain_metric_short.txt', 'r') as f:
        explain_metric = f.read()
        
    with open('prompt/output_format.txt', 'r') as f:
        output_format = f.read()
    
    system_prompt = f"""
    You are an English teaching assistant, and you are good at grading essays, and your students need you for their IELTS Academic essay task 2. You will be given the topic and student's response.
    You should grade the essay general score and in 4 metrics in IELTS Writing, which are `Task Response`, `Coherence and Cohesion`, `Lexical Resource` and `Grammatical Range and Accuracy`. 
    The overall score can be a float between 0 and 9 (round to .5), but each metric score should be an integer between 0 and 9.

    Recall the IELTS Writing band score criteria

    Here are the criteria for each metric at each band score:
    {explain_metric}

    In each metric, you should give a detailed explanation and point out exactly the student mistakes that led to that score.
    Your output format should be like this:
    {output_format}

    - The overall score should be the mean value of 4 metric scores, round down to .0 and .5, so make sure your evaluation right.
    - Provide constructive feedback.
    """
    return system_prompt


def get_instruction_prompt(essay_topic, student_response):
    instruction_prompt_1 = f"""
    You will have to grade this essay in IELTS WRITING task 2 academic guideline
    Here is the topic of the essay: {essay_topic}
    And here is my essay: {student_response}
    Make sure that you follow the IELTS WRITING task 2 guideline to grade this essay, and do the correct evaluation of general score.
    """
    return instruction_prompt_1


def incorrect_data(pipe, messages, general, tr, cc, lr, gr, **generation_args):
    if (tr + cc + lr + gr)//2 != general * 2:
        print("Inconsistent scoring")
        proposed_correct_score = ((tr + cc + lr + gr+1)//2)/2
        add_prompt = {"role": "user", "content": f"Your score doesn't make sense. How can I get a {general} in general while I only get {tr} in Task Response, {cc} in Coherence and Cohesion, {lr} in Lexical Resource and {gr} in Grammatical Range and Accuracy, which the final result is {proposed_correct_score}? You need to check the grade of my essay again, and maintain the output format"},
        messages.append(add_prompt[0])
        output = pipe(messages, **generation_args["model_args"])
        messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
        return output[0]['generated_text']
    else:
        return messages[-1]['content']


def incorrect_data2(pipe, messages, general, tr, cc, lr, gr, **generation_args):
    if (tr + cc + lr + gr)//2 != general * 2:
        print("Inconsistent scoring")
        proposed_correct_score = ((tr + cc + lr + gr+1)//2)/2
        add_prompt = {"role": "user", "content": f"Your score is not consistence and still does not sum up equally. You score this essay {tr} in Task Response, {cc} in Coherence and Cohesion, {lr} in Lexical Resource and {gr} in Grammatical Range and Accuracy, so the total score should be {proposed_correct_score}, which is different than your original score of {gr}. You should grade my essay again, and maintain the output format"},
        messages.append(add_prompt[0])
        output = pipe(messages, **generation_args["model_args"])
        messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
        return output[0]['generated_text']
    else:
        return messages[-1]['content']

def rescore(pipe, messages, mode, **generation_args):
    adj = 'strict' if mode == 'harsh' else 'loose'
    messages.append({"role": "user", "content": f"Do you think you are too {mode}? Your explanation for each metric seems to be too {adj} and may not fit with its criteria. You should reevaluate your score and remain the output format and make sure all the criteria scores is integer and only general score can be float (round to .5)."})
    
    output = pipe(messages, **generation_args["model_args"])
    messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
    return output[0]['generated_text']


def indepth_feedback(pipe, messages, **generation_args):
    messages.append({"role": "user", "content": f"Provide more detailed feedback in the essay, including all the mistake made in the essay and how to improve it."})
    output = pipe(messages, **generation_args["model_args"])
    messages.append({"role": generation_args["role"], "content": output[0]['generated_text']},)
    return output[0]['generated_text']


def until_correct(pipe, messages, **generation_args):
    loop_count = 0
    general, tr, cc, lr, gr = get_score(messages[-1]['content'])
    result = messages[-1]['content']
    while (tr + cc + lr + gr)//2 != general*2 or int(tr) != tr or int(cc) != cc or int(lr) != lr or int(gr) != gr:
        loop_count+=1
        print(f"______Adjusting score {loop_count}_______")
        if loop_count%2 == 0:
            result = incorrect_data2(pipe, messages, general, tr, cc, lr, gr, **generation_args)
        else:
            result = incorrect_data(pipe, messages, general, tr, cc, lr, gr, **generation_args)
        
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