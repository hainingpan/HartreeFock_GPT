import yaml

directory='../'
def generate_prompt(template,docstring,paper,trial_idx=0,directory=directory,save=True):
    with open(directory+template, 'r') as f:
        prompt = f.read()
    
    with open(directory+docstring, 'r') as f:
        docstring = f.read()
    
    with open(directory+'ground_truth.yaml', 'r') as f:
        gt = yaml.load(f, Loader=yaml.FullLoader)
    for val in gt:
        if val['arxiv'] == float(paper):
            hamiltonian = val['gt']
            symmetry = val['symmetry']
            break
    
    prompt = prompt.replace("{{docstring}}", '\n'+docstring).replace("{{hamiltonian}}", '\n'+hamiltonian).replace("{{symmetry}}", symmetry)

    output_fn= 'prompt_{int}_{decimal}_{trial_idx}'.format(int=paper.split('.')[0],decimal=paper.split('.')[1],trial_idx=trial_idx) + '.md' 
    if save:
        with open(output_fn, 'w') as f:
            f.write(prompt)
    return prompt

### LLM-Generated code
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import anthropic
client = anthropic.Anthropic()
def code_generate(prompt,  client=client, model='claude-3-7-sonnet-20250219', max_tokens=12800,budget_tokens=6400,verbose=True):
    messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    thinking={
        "type": "enabled",
        "budget_tokens": budget_tokens
    }
    results = {"thinking": "", "text": ""}
    with client.messages.stream(model = model,max_tokens=max_tokens, thinking=thinking, messages=messages) as stream:
        current_block_type = None 
        thinking_flag = True
        text_flag = True
        for event in stream:
            if event.type == "content_block_start":
                current_block_type = event.content_block.type
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    results["thinking"] += event.delta.thinking
                    if thinking_flag and verbose:
                        print()
                        print("#"*20,'THINKING','#'*20,  flush=True)
                        print()
                        thinking_flag = False
                    print(event.delta.thinking, end='', flush=True)
                elif event.delta.type == "text_delta":
                    results["text"] += event.delta.text   
                    if text_flag and verbose:
                        print('\n')
                        print("#"*20,'TEXT','#'*20,  flush=True)
                        print()
                        text_flag = False   
                    print(event.delta.text,end='', flush=True)     
            elif event.type == "message_stop":
                break
    return results


import re
def extract_code(code_response):
    # get python code. WARNING! make sure code is surrounded by ```
    matches = re.findall(r'```(.*?)```', code_response, re.DOTALL)
    proc_codes = []
    for match in matches:
        a = match.split('\n')
        if a[0].lower() == 'python':
            code = '\n'.join(a[1:])
        else:
            code = '\n'.join(a)
        proc_codes.append(code)
    python_code = '\n'.join(proc_codes)
    return python_code

def save_code(code,paper,trial_idx):
    output_fn= 'code_{int}_{decimal}_{trial_idx}'.format(int=paper.split('.')[0],decimal=paper.split('.')[1],trial_idx=trial_idx) + '.py'
    with open(output_fn, 'w') as f:
        f.write(code)
        print(f"Code saved to {output_fn}")
### Code evalution
import sys
sys.path.append('../')
import matplotlib.pyplot as plt

import HF
import numpy as np
import inspect

def plot_kspace(kspace):
    fig, ax = plt.subplots(figsize=(3,3),tight_layout=True)
    ax.scatter(*kspace.T,s=2)
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
def plot_matele(mat):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.spy((mat))
    ax.set_title('$H_0$')
def plot_2d_bandstructure(ham,en):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(en.shape[0]):
        ax.plot_trisurf(ham.k_space[:,0],ham.k_space[:,1],en[i])
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_zlabel('E')
def plot_high_symm_bandstructure(k_list,en,ax=None): # DROP
    if ax is None:
        fig, ax = plt.subplots()
    for e in en:
        k_abs=np.sqrt(np.diff(k_list[:,0])**2 + np.diff(k_list[:,1])**2)
        k_abs = np.concatenate([[0],np.cumsum(k_abs)])
        ax.plot(k_abs,e,color='k')
    Nk = (k_list.shape[0]-1)//4
    for i in range(5):
        ax.axvline(k_abs[Nk*i],ls='--',color='r')
    ax.set_xticks([k_abs[Nk*i] for i in range(5)],['$\Gamma$','K','M','$\Gamma$',"K'"])
def plot_2d_false_color_map(ham,en,kmax=4,width= 3):
    kmax = min(len(en),kmax)   # number of bands
    
    fig, ax = plt.subplots(1,kmax,figsize=(3*kmax,3),tight_layout=True)
    for idx in range(kmax):
        im=ax[idx].tripcolor(ham.k_space[:,0],ham.k_space[:,1],en[idx],shading='gouraud')
        ax[idx].set_title(f'Band {idx+1}')
        ax[idx].set_xlabel('$k_x$')
        ax[idx].set_ylabel('$k_y$')
        ax[idx].set_aspect('equal')
        plt.colorbar(im, ax=ax[idx],)
    
    

def print_gap(ham_int,exp_val,en_int):
    mean_U=np.abs(ham_int.generate_interacting(exp_val)).mean() 
    mean_T=np.abs(ham_int.generate_non_interacting()).mean() 
    levels = len(en_int)
    gap = en_int[levels//2].min()-en_int[levels//2-1].max()
    print(f'Gap is {gap:.2f}')
    print(f'U/T is {mean_U/mean_T:.2f}')
    print(f'mean_U is {mean_U:.2f}')    