"""Adapted utils.py to perform the execution step with Palm."""
import google.generativeai as palm
import argparse
import json
import os
import yaml

class PalmExecution:
    def __init__(self, api_key_dict, prompt_template, arxiv_number, model='text-bison-001', temperature=0):
        api_key = api_key_dict['API_KEY']
        palm.configure(api_key)
        self.prompt_template = prompt_template
        self.arxiv_number = arxiv_number
        self.model_config = {'model': model, 'temperature': temperature}

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

    def generate_prompt(kwargs, prompt_dict):
        if 'task' in kwargs:
            correct_phdict=return_correct_prompt_template_for_task(kwargs)
            content=prompt_dict[kwargs['task']].format(**correct_phdict).strip()
            for key,val in correct_phdict.items():
                if isinstance(val, bool):
                    key_=key.format(**correct_phdict)
                    if val:
                        content=content.replace('['+key_+']',key_)
                    else:
                        content=content.replace('['+key_+']','')
            message = {
                'role': 'user',
                'content': content,
            }
            return message

    def return_correct_prompt_template_for_task(task):
      """Parses a single entry in the yaml file for a paper to construct the correct (ground truth) completed template as a dict of placeholder->entry."""
      correct_phdict = {}
      for ph in task['placeholder']:
        if (task['placeholder'][ph]['human']) is not None: # LLM was wrong
          correct_phdict.update({ph: task['placeholder'][ph]['human']})
        else:
          if task['placeholder'][ph]['score']['Haining']==2:
            correct_phdict.update({ph: task['placeholder'][ph]['LLM']})
          else:
            raise ValueError(f'Omitting Task {task}: No correct answer')
      return correct_phdict

    def assembly_message(sys_msg,user_msg,AI_msg):
        messages = sys_msg
        assert len(user_msg)-len(AI_msg)==1, f'# of user message {len(user_msg)} is not compatible with # of AI_message {len(AI_msg)}'
        messages.append(user_msg[0])
        for user, AI in zip(user_msg[1:],AI_msg):
            messages.append(AI)
            messages.append(user)
        return messages

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
        rs = palm.generate_text(summarization_prompt, self.model_config)
        summarized=rs.result

        if len(var)>0:
            if 'Use the following conventions for the symbols:  ' in summarized:
                summarized += '\n'+'\n'.join(var)
            else:
                summarized += '\n\nUse the following conventions for the symbols:  \n' +'\n'.join(var)
        return summarized

    def solver(summarization, prompt, prompt_dict):
        '''
        Solve the problem in the prompt
        '''
        sys_msg=prompt_dict['Problem-solver']
        question_prompt='**Background**  \n{background}\n\n**Question**  \n{question}'.format(background=summarization,question=prompt)
        messages = sys_msg + '\n'+ question_prompt
        rs= palm.generate_text(prompt=messages, **self.model_config)
        return rs.result


    def run(prompt_template, arxiv_number):
        '''Load the prompt_template, and the descriptor file from arxiv number
        Generate prompts, and feed into `solver`.
        The response will be summarized by `summarizer`.
        Write all responses to `{arxiv_number}_auto.md`

        Should run from each directory 'arxiv_number'.'''
        prompt_dict=load_prompt_template(prompt_template)
        with open(f'{arxiv_number}.yaml','r') as f:
            kwargs= yaml.safe_load(f)
        kwargs=[kwarg for kwarg in kwargs if 'task' in kwarg]

        prompts=[generate_prompt(kwarg,prompt_dict=prompt_dict) for kwarg in kwargs]

        answers=[]
        for idx,prompt_i in enumerate(prompts):
            print(f'Asking {idx}..')
            prompt=prompt_i['content']
            if idx==0:
                summarization=''
                response=solver(summarization=summarization, prompt=prompt,prompt_dict=prompt_dict)
            else:
                summarization=summarizer(summarization=summarization, prompt=prompt, response=response,prompt_dict=prompt_dict)        
                response=solver(summarization=summarization, prompt=prompt,prompt_dict=prompt_dict)
            answers.append(response)
        
        string=''
        for kwarg,prompt_i,answer in zip(kwargs,prompts,answers):
            task=kwarg['task']
            prompt=prompt_i['content']
            added=f'## {task}  \n**Prompt:**  \n{prompt}\n\n**Completion:**  \n{answer}\n\n'
            string+=added

        with open(f'{arxiv_number}_auto.md','w') as f:
            f.write(string)

