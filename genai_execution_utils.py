"""Adapted utils.py to perform the execution step with GenAI."""
import google.generativeai as genai
import argparse
import json
import os
import yaml

from . import utils
# from utils import load_prompt, generate_prompt, return_correct_prompt_template_for_task, assembly_message, extract_var

class StreamlinedExecution:
    def __init__(self, api_key_dict, model_name='gemini-pro', temperature=0):
        api_key = api_key_dict['API_KEY']
        genai.configure(api_key = api_key)
        self.model_name = model_name
        self.generation_config = {'temperature': temperature, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 2048, 'stop_sequences': []}
        self.safety_settings = [{'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}]
        self.model = genai.GenerativeModel(model_name=model_name)
        self.history = []
    
    def update_history(self, input, response):
        self.history.append(("USER: "+input, "ASSISTANT: "+response))

    def render_history(self):
        chat = ''
        for prompt, reponse in self.history:
            chat += prompt
            chat += ('\n'+response+'\n\n')
        return chat

    def run(self, prompt_dict):
        '''Concatenates the full list of past prompts and responses.'''
        
        for key, prompt in prompt_dict:
            print(f'Asking {idx}..', prompt)
            print('####')
            execution_prompt = self.render_history() + prompt
            response = self.model.generate_content(execution_prompt, generation_config=self.generation_config, safety_settings=self.safety_settings, stream=False)
            print(response.text)
            print('===\n===')
            self.update_history(prompt, reponse.text)
        
        string=''
        for kwarg,tup in zip(kwargs, self.history):
            task=kwarg['task']
            prompt = tup[0]
            reply = tup[1]
            added=f'## {task}  \n**Prompt:**  \n{prompt}\n\n**Completion:**  \n{reply}\n\n'
            string+=added

        with open(f'{arxiv_number}_auto_gemini.md','w') as f:
            f.write(string)


class PalmExecution:
    def __init__(self, api_key_dict, prompt_template, arxiv_number, model_name='text-bison-001', temperature=0):
        api_key = api_key_dict['API_KEY']
        genai.configure(api_key = api_key)
        self.prompt_template = prompt_template
        self.arxiv_number = arxiv_number
        self.model_name = model_name
        self.generation_config = {'temperature': temperature, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 2048, 'stop_sequences': []}
        self.safety_settings = [{'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'},
        {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}]
        self.model = genai.GenerativeModel(model_name=model_name)

    def summarizer(self, summarization, prompt, response,prompt_dict):
        '''Summarize the background (summarization) + question (prompt) + answer (response)'''
        var_old= utils.extract_var(summarization)
        var_new= utils.extract_var(prompt)
        var=var_old+var_new
        
        summarization_prompt=prompt_dict['Conversation summarizer'].format(background=summarization, question=prompt, answer=response)
        rs = self.model.generate_content(summarization_prompt, generation_config=self.generation_config,
            safety_settings=self.safety_settings, stream=False)
        summarized=rs.text

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
        execution_prompt = sys_msg + '\n'+ question_prompt
        rs= self.model.generate_content(execution_prompt, generation_config=self.generation_config,
            safety_settings=self.safety_settings, stream=False)
        return rs.text

    def run(self, prompt_template, arxiv_number, path='HartreeFock_GPT/'):
        '''Load the prompt_template, and the descriptor file from arxiv number
        Generate prompts, and feed into `solver`.
        The response will be summarized by `summarizer`.
        Write all responses to `{arxiv_number}_auto.md`

        Should run from each directory 'arxiv_number'.'''
        prompt_dict= utils.load_prompt_template(prompt_template)
        with open(os.path.join(path, f'{arxiv_number}/{arxiv_number}.yaml'),'r') as f:
            kwargs= yaml.safe_load(f)
        kwargs=[kwarg for kwarg in kwargs if 'task' in kwarg]

        prompts=[utils.generate_prompt(kwarg,prompt_dict=prompt_dict) for kwarg in kwargs]

        answers=[]
        summaries = []
        for idx,prompt_i in enumerate(prompts):
            print(f'Asking {idx}..')
            prompt=prompt_i['content']
            if idx==0:
                summarization=''
                response= self.solver(summarization=summarization, prompt=prompt,prompt_dict=prompt_dict)
            else:
                summarization=self.summarizer(summarization=summarization, prompt=prompt, response=response,prompt_dict=prompt_dict)        
                response= self.solver(summarization=summarization, prompt=prompt,prompt_dict=prompt_dict)
            answers.append(response)
            summaries.append(summarization)
        
        string=''
        for kwarg,prompt_i,answer in zip(kwargs,prompts,answers):
            task=kwarg['task']
            prompt=prompt_i['content']
            added=f'## {task}  \n**Prompt:**  \n{prompt}\n\n**Completion:**  \n{answer}\n\n'
            string+=added

        with open(f'{arxiv_number}_auto_gemini.md','w') as f:
            f.write(string)

        # With the summarizer
        string=''
        for kwarg,prompt_i,answer,summary in zip(kwargs,prompts,answers,summaries):
            task=kwarg['task']
            prompt=prompt_i['content']
            added=f'## {task}  \n**Background:**  \n{summary}\n**Prompt:**  \n{prompt}\n\n**Completion:**  \n{answer}\n\n'
            string+=added

        with open(f'{arxiv_number}_auto_withsummarizer_gemini.md','w') as f:
            f.write(string)

