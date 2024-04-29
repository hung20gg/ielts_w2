import re

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

def clean_output(output_string):
    output_part = output_string.split('**')
    out2 = '**' + '**'.join(output_part[1:11])
    return out2


def get_score(output):
    general = 0
    tr = 0
    cc = 0
    lr = 0
    gr = 0
    match = re.search(r'General\s?:\s?((?<!\d)\d+(?:\.\d+)?)', output)
    if match:
        general = float(match.group(1))
    match = re.search(r'Task Response\s?:\s?((?<!\d)\d+(?:\.\d+)?)', output)
    if match:
        tr = float(match.group(1))
        
    match = re.search(r'Coherence and Cohesion\s?:\s?((?<!\d)\d+(?:\.\d+)?)', output)
    if match:
        cc = float(match.group(1))
        
    match = re.search(r'Lexical Resource\s?:\s?((?<!\d)\d+(?:\.\d+)?)', output)
    if match:
        lr = float(match.group(1))
    
    match = re.search(r'Grammatical Range and Accuracy\s?:\s?((?<!\d)\d+(?:\.\d+)?)', output)
    if match:
        gr = float(match.group(1))
        
    assert general * tr * cc * lr * gr != 0, "Can't extract score from output"
    
    return general, tr, cc, lr, gr


def get_essay(response):
    response_chunks = response.split('**')
    accept_flags = ['revised essay', 'rewrite essay', 'modified essay', 'adjusted essay']
    
    essay = ''
    for i in range(len(response_chunks) - 1):
        if response_chunks[i].lower().replace(':','') in accept_flags:
            essay = response_chunks[i+1]
            
    if len(essay) == 0:
        return essay
    return essay.split('Note:')[0]