import json
import os
import yaml
import re

def get_task_yaml_for_paper(arxiv_id, path):
  path = os.path.join(path, arxiv_id + '.yaml')
  paper_tasks = yaml.safe_load(open(path, 'r'))
  return paper_tasks

def load_excerpt(subdir, sources):
    excerpt=''
    for tex, lines in sources.items():
        with open(os.path.join(subdir, tex),'r') as f:
            f_list=list(f)
            for line in lines:
              if line: #sometimes it's []
                excerpt=excerpt+''.join(f_list[line[0]:line[1]])
    return excerpt

def return_correct_prompt_template_for_task(task):
  # Parses a single entry in the yaml file for a paper to construct the correct (ground truth) completed template
  correct_phdict = {}
  for ph in task['placeholder']:
    if bool(task['placeholder'][ph]['human']): # LLM was wrong
      correct_phdict.update({ph: task['placeholder'][ph]['human']})
    else:
      if task['placeholder'][ph]['score']['Haining']==2:
        correct_phdict.update({ph: task['placeholder'][ph]['LLM']})
      else:
        print(f'Omitting Task {task}: No correct answer')
  return correct_phdict

def fill_placeholders(placeholders, placeholders_optional, mapping, empty_template):
  for ph in placeholders:
    tag = '{'+ph+'}'
    if tag not in empty_template:
      print(f'Error: No placeholder {tag} in template')
    else:
      try:
        empty_template = empty_template.replace(tag, mapping[ph])
      except KeyError as e:
        print(f'Placeholder {tag} not in Yaml File.')
  for ph in placeholders_optional:
    tag = '['+ph+']'
    if tag not in empty_template:
      print(f'Error: No placeholder {tag} in template')
    else:
      try:
        empty_template = empty_template.replace(f'[{ph}]', str(mapping[ph]))
      except KeyError as e:
        print(f'Placeholder {tag} not in Yaml File.')
  return empty_template

#from Haining score.ipynb notebooks
def extract_filled_values(template_str):
  # Returns all mandatory and optional placeholders for a template
  template_str = template_str.replace('{{', '').replace('}}', '')
  # Extract placeholders from the template
  # placeholders = re.findall(r"\{(\w+)\}", template_str)
  placeholders = re.findall(r"[\{]([\w\|\-$ ,.\{\}]+?)[\}]", template_str)
  placeholders_optional = re.findall(r"[\[]([\w\|\-$ ,.\{\}]+)[\]]", template_str)
  # Create a regex pattern to match the filled values based on the placeholders
  placeholders = list(set(placeholders))
  placeholders_optional = list(set(placeholders_optional))
  return placeholders, placeholders_optional

def make_df_for_paper(gdirpath, arxiv_id, task_templates):
  print(f'Arxiv Id: {arxiv_id}  ##############')
  paper_dir = os.path.join(gdirpath, arxiv_id)
  paper_yaml = get_task_yaml_for_paper(arxiv_id, paper_dir)
  tasks, templates, excerpts, gt_mapping, annotated_prompts, annotation_status = [], [], [], [], [], []
  for elem in paper_yaml:
    if 'task' in elem:
      task_name = elem['task']
      task_template = task_templates[task_name]
      print(f"Task {task_name}")
      tasks.append(task_name)
      templates.append(task_template)
      excerpt = load_excerpt(paper_dir, elem['source'])
      excerpts.append(excerpt)

      # get placeholder GT mapping
      phdict = return_correct_prompt_template_for_task(elem)
      annotation_status.append(bool(phdict))
      placeholders, placeholders_opt = extract_filled_values(task_template) # placeholder {} | placeholder_optional [] list
      completed = fill_placeholders(placeholders, placeholders_opt, phdict, task_template)
      gt_mapping.append(json.dumps(phdict))
      annotated_prompts.append(completed)
  
  return pd.DataFrame({'arxiv_id': [arxiv_id]*len(tasks), 'task': tasks, 'excerpt': excerpts, 'blank_templates': templates, 'gt_mapping': gt_mapping, 'annotated_prompts': annotated_prompts, 'annotated': annotation_status})


