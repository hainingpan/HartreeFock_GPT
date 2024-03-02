import yaml
import argparse
from utils import load_prompt_template
from functools import partial
prompt_system='''I will provide you a Excerpt of physics paper, and a Template. You will need to fill the placeholders in the template using the correct information from the excerpt.
Here are conventions:  
{..} means a placeholder which you need to fill by extracting information from the excerpt.  
{A|B} means you need to make a choice between A and B  
[..] means optional sentence. You should decide whether to use it depending on the excerpt.  
{{..}} DOES NOT mean a placeholder. You should not change the content inside double curly braces {{..}}.  
'You should recall that {..}.' : this sentence should be kept as is.

Finally, if you cannot figure out the placeholder, you should leave it as is.
'''
def drop_text_after(string):
    string_new=''
    for line in string.split('\n'):
        if '===' in line:
            break
        else:
            if 'You should recall that' not in line:
                string_new='\n'.join([string_new,line])
    return string_new[1:]

def load_excerpt(sources,arxiv_number):
    excerpt=''
    for tex, lines in sources.items():
        with open(arxiv_number+'/'+tex,'r') as f:
            f_list=list(f)
            for line in lines:
                excerpt=excerpt+''.join(f_list[line[0]:line[1]])
    return excerpt

prompt_dict = partial(load_prompt_template,file='prompt_template.md')

def extractor(descriptor,arxiv_number,message=False):
    sys_msg=[{'role': 'system', 'content': prompt_system}]
    question_prompt='\nTemplate:\n {template} \n\n Excerpt:\n {excerpt}'.format(template=drop_text_after(prompt_dict()[descriptor['task']]), excerpt=load_excerpt(descriptor['source'],arxiv_number))
    user_msg=[{'role':'user','content':question_prompt}]
    messages = sys_msg + user_msg
    if message:
        return sys_msg[0]['content']+user_msg[0]['content']
    rs= openai.ChatCompletion.create(messages=messages, **model_params)
    response=rs['choices'][0]['message'].content
    return response
      


def main():
    parser = argparse.ArgumentParser(description='Run problem solving with AI based on given template and Arxiv paper.')
    # parser.add_argument('prompt_template', type=str, help='Path to the prompt template file.')
    parser.add_argument('arxiv_number', type=str, help='Arxiv paper number.')
    # parser.add_argument('--interactive', action='store_true', help='Whether to pause after each task.')

    args = parser.parse_args()
    with open(args.arxiv_number+'/'+args.arxiv_number+'.yaml','r') as f:
        kwargs_yaml = yaml.safe_load(f)
    

    with open('Naming.yaml','r') as f:
        naming=yaml.safe_load(f)


    with open(args.arxiv_number+'/'+args.arxiv_number+'_extractor.md','r') as f:
        completions=[]
        new_name=''
        for line in f:
            if line.startswith('#'):
                if new_name!='':
                    completions.append((new_name,contents))
                task_name=line.replace('#','').strip()
                new_name=task_name
                contents=''
            else:
                contents+=line
        completions.append((new_name,contents))  

    string=''
    for kwargs,(name,completion) in zip(kwargs_yaml[1:],completions):
        print(name)
        assert name==kwargs['task'], f"completion name {name} and yaml task {kwargs['task']} do not match"
        string+='# '+naming[name]+'\n'
        response=(extractor(kwargs, args.arxiv_number, message=True))
        string+='**Prompt**  \n'+response+'\n\n'
        string+='**Completion**  \n'+completion+'\n\n'
    
    with open(args.arxiv_number+'/'+args.arxiv_number+'_extractor_example.md','w') as f:
        f.write(string)

if __name__ == "__main__":
    main()

# To run this script, use the following exemplary command:
# python combine_extractor.py 2111.01152