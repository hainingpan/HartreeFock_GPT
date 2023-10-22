"""Adapted utils.py to perform the execution step with Palm."""
import google.generativeai as palm
import argparse
import json
import os
import yaml

from . import utils
# from utils import load_prompt, generate_prompt, return_correct_prompt_template_for_task, assembly_message, extract_var

class PalmExecution:
    def __init__(self, api_key_dict, prompt_template, arxiv_number, model='text-bison-001', temperature=0):
        api_key = api_key_dict['API_KEY']
        palm.configure(api_key = api_key)
        self.prompt_template = prompt_template
        self.arxiv_number = arxiv_number
        self.model_config = {'model': model, 'temperature': temperature}

    def summarizer(self, summarization, prompt, response,prompt_dict):
        '''Summarize the background (summarization) + question (prompt) + answer (response)'''
        var_old= utils.extract_var(summarization)
        var_new= utils.extract_var(prompt)
        var=var_old+var_new
        
        summarization_prompt=prompt_dict['Conversation summarizer'].format(background=summarization, question=prompt, answer=response)
        rs = palm.generate_text(summarization_prompt, self.model_config)
        summarized=rs.result

        if len(var)>0:
            if 'Use the following conventions for the symbols:  ' in summarized:
                summarized += '\n'+'\n'.join(var)
            else:
                summarized += '\n\nUse the following conventions for the symbols:  \n' +'\n'.join(var)
        return summarized

    def solver(self, summarization, prompt, prompt_dict):
        '''
        Solve the problem in the prompt
        '''
        sys_msg=prompt_dict['Problem-solver']
        question_prompt='**Background**  \n{background}\n\n**Question**  \n{question}'.format(background=summarization,question=prompt)
        messages = sys_msg + '\n'+ question_prompt
        rs= palm.generate_text(prompt=messages, **self.model_config)
        return rs.result

    def run(self, prompt_template, arxiv_number):
        '''Load the prompt_template, and the descriptor file from arxiv number
        Generate prompts, and feed into `solver`.
        The response will be summarized by `summarizer`.
        Write all responses to `{arxiv_number}_auto.md`

        Should run from each directory 'arxiv_number'.'''
        prompt_dict= utils.load_prompt_template(prompt_template)
        with open(f'{arxiv_number}/{arxiv_number}.yaml','r') as f:
            kwargs= yaml.safe_load(f)
        kwargs=[kwarg for kwarg in kwargs if 'task' in kwarg]

        prompts=[utils.generate_prompt(kwarg,prompt_dict=prompt_dict) for kwarg in kwargs]

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

        with open(f'{arxiv_number}_auto_palm.md','w') as f:
            f.write(string)

