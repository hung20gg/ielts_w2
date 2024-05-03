import re

def get_output_suggestion_format():
    with open('prompt/output_suggestion_format.txt', 'r') as f:
        output_suggestion_format = f.read()
    return output_suggestion_format


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