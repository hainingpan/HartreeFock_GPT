import argparse
import json
import openai
from dotenv import load_dotenv
import os
import yaml

model_params = {
    'model': "gpt-4",
    'temperature': 0,
}

load_dotenv('../../.env')
openai.api_key  = os.getenv('OPENAI_API_KEY')

def load_prompt_template(file):
    prompt_dict = {}
    current_task = None  # Track the current task, initialized to None

    with open(file, 'r') as f:
        for line in f:
            stripped_line = line.strip()

            # Skip comments
            if stripped_line.startswith('<!--'):
                continue
            
            if stripped_line.startswith('## '):
                # New task starts
                current_task = stripped_line[3:]
                prompt_dict[current_task] = ''
            elif not stripped_line.startswith('# ') and current_task:
                # If this line is part of a task (not a task declaration or a section),
                # and a task is currently being tracked, append this line to the task.
                if not stripped_line.startswith('**Prompt:**'):
                    prompt_dict[current_task] += line

    return prompt_dict

def generate_prompt(content):
    message = {
        'role': 'user',
        'content': content,
    }
    return message

# def return_correct_prompt_template_for_task(task):
#   """Parses a single entry in the yaml file for a paper to construct the correct (ground truth) completed template as a dict of placeholder->entry."""
#   correct_phdict = {}
#   for ph in task['placeholder']:
#     if (task['placeholder'][ph]['human']) is not None: # LLM was wrong
#       correct_phdict.update({ph: task['placeholder'][ph]['human']})
#     else:
#       if task['placeholder'][ph]['score']['Haining']==2:
#         correct_phdict.update({ph: task['placeholder'][ph]['LLM']})
#       else:
#         raise ValueError(f'Omitting Task {task["task"]}/{ph} No correct answer')
#   return correct_phdict

# def assembly_message(sys_msg,user_msg,AI_msg):
#     messages = sys_msg
#     assert len(user_msg)-len(AI_msg)==1, f'# of user message {len(user_msg)} is not compatible with # of AI_message {len(AI_msg)}'
#     messages.append(user_msg[0])
#     for user, AI in zip(user_msg[1:],AI_msg):
#         messages.append(AI)
#         messages.append(user)
#     return messages

def extract_var(prompt):
    string='Use the following conventions for the symbols'
    contains=False
    var_list=[]
    for line in prompt.split('\n'):
        if contains:
            if len(line.strip())==0:
                break
            var_list.append(line)

        if line.startswith(string):
            contains=True
        
    return var_list

def summarizer(summarization, prompt, response,prompt_dict):
    '''Summarize the background (summarization) + question (prompt) + answer (response)'''
    var_old=extract_var(summarization)
    var_new=extract_var(prompt)
    var=var_old+var_new
    
    summarization_prompt=prompt_dict['Conversation summarizer'].format(background=summarization,question=prompt, answer=response)
    messages= [{'role':'user','content': summarization_prompt}]
    rs= openai.ChatCompletion.create(messages=messages, **model_params)

    summarized=rs['choices'][0]['message'].content

    
    if len(var)>0:
        if 'Use the following conventions for the symbols:  ' in summarized:
            summarized += '\n'+'\n'.join(var)
        else:
            summarized += '\n\nUse the following conventions for the symbols:  \n' +'\n'.join(var)
    return summarized

def solver(summarization, prompt,prompt_dict):
    '''
    Solve the problem in the prompt
    '''
    sys_msg=[{'role': 'system', 'content': prompt_dict['Problem-solver']}]
    question_prompt='**Background**  \n{background}\n\n**Question**  \n{question}'.format(background=summarization,question=prompt)
    user_msg=[{'role':'user','content':question_prompt}]
    messages = sys_msg + user_msg
    rs= openai.ChatCompletion.create(messages=messages, **model_params)

    response=rs['choices'][0]['message'].content
    return response

# def load_yaml(arxiv_number):
#     # Repetative but ensure yaml is always latest even if I change the yaml after start running 
#     with open(f'{arxiv_number}.yaml','r') as f:
#         kwargs= yaml.safe_load(f)
#     return [kwarg for kwarg in kwargs if 'task' in kwarg] # remove the branch of this file

def read_markdown(filename):
    prompt_dict={}
    prompt=''
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('###'):
                prompt=''
            elif line.startswith('##'):
                if len(prompt) >0:
                    prompt_dict[(task,subtask)]=prompt
                    prompt=''
                subtask=line.replace('##','').strip()
            elif line.startswith('#'):
                if len(prompt) >0:
                    prompt_dict[(task,subtask)]=prompt
                    prompt=''
                task=line.replace('#','').strip()
            else:
                prompt+=line
        prompt_dict[(task,subtask)]=prompt
    return prompt_dict
    
def run(prompt_template,prompt_file,interactive):
    '''Load the prompt_template, and the descriptor file from arxiv number
    Generate prompts, and feed into `solver`.
    The response will be summarized by `summarizer`.
    Write all responses to `{arxiv_number}_auto.md`

    Should run from each directory 'arxiv_number'.'''
    prompt_template_dict=load_prompt_template(prompt_template)
    prompt_content_dict=read_markdown(prompt_file)
    writepath=f'{os.path.split(prompt_file)[-1].replace(".md","")}_auto.md'
    if os.path.exists(writepath):
        raise ValueError(f'{writepath} already exists')
    
    pre_task=''
    idx=0
    for (task,subtask), prompt in prompt_content_dict.items():
        prompt_i=generate_prompt(content=prompt)
        print(f'Asking {task}/{subtask}..')
        if idx==0:
            summarization=''
            response=solver(summarization=summarization, prompt=prompt,prompt_dict=prompt_template_dict)
        else:
            summarization=summarizer(summarization=summarization, prompt=prompt, response=response,prompt_dict=prompt_template_dict)        
            response=solver(summarization=summarization, prompt=prompt,prompt_dict=prompt_template_dict)

        string=''
        if task!=pre_task:
            string+=f'# {task}\n'
            pre_task=task
        string+=f'## {subtask}\n'
        string+=f'###\n**Prompt:**  \n{prompt}\n\n###\n**Completion:**  \n{response}\n\n'
        with open(writepath,'a') as f:
            f.write(string)

        if interactive:
            input('Press Enter to continue...')

def main():
    parser = argparse.ArgumentParser(description='Run problem solving with AI based on given prompt file.')
    parser.add_argument('prompt_template', type=str, help='Path to the prompt template file.',default='prompt_template.md')
    parser.add_argument('prompt', type=str, help='Path to the prompt file.',default='2111.01152_0/prompt.md')
    # parser.add_argument('arxiv_number', type=str, help='Arxiv paper number.')
    parser.add_argument('--interactive', action='store_true', help='Whether to pause after each task.')

    args = parser.parse_args()

    # run(args.arxiv_number, args.interactive)
    run(args.prompt_template,args.prompt,args.interactive)

if __name__ == "__main__":
    main()

# To run this script, use the following exemplary command:
# python ../utils_md.py ../Prompt_template.md prompts.md 