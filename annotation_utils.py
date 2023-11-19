import json
import os
import yaml
import re

import pandas as pd
import numpy as np

# constants
ANNOTATED_PAPERS = ['1010.1819', '1106.6060', '1208.0116', '1310.2674', '1812.04213', '2004.04168', '2008.08998', '2012.04554', '2108.02159', '2110.11330', '2111.01152', '2112.07523', '2308.03843', '2308.07488']
# , '1212.5363''0812.2894', '0902.1336', 
def get_task_yaml_for_paper(arxiv_id, path):
  """Returns the scored yaml task file for a specific paper."""
  path = os.path.join(path, arxiv_id + '.yaml')
  paper_tasks = yaml.safe_load(open(path, 'r'))
  return paper_tasks

# Adapted from the extractor.ipynb notebooks. 
def load_excerpt(subdir, sources):
  """Load the excerpt for a given paper."""
  excerpt=''
  for tex, lines in sources.items():
      with open(os.path.join(subdir, tex),'r') as f:
          f_list=list(f)
          for line in lines:
            if line: #sometimes it's []
              excerpt=excerpt+''.join(f_list[line[0]:line[1]])
  return excerpt

def return_correct_prompt_template_for_task(task):
  """Parses a single entry in the yaml file for a paper to construct the correct (ground truth) completed template as a dict of placeholder->entry."""
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
  """Returns the annotated prompt."""
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

# Adapted from Haining score.ipynb notebooks
def extract_filled_values(template_str):
  """Returns all mandatory and optional placeholders for a template."""
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
  eg_tag = '\n==='
  paper_dir = os.path.join(gdirpath, arxiv_id)
  paper_yaml = get_task_yaml_for_paper(arxiv_id, paper_dir)
  tasks, templates, excerpts, gt_mapping, annotated_prompts, annotation_status = [], [], [], [], [], []
  for elem in paper_yaml:
    if 'task' in elem:
      task_name = elem['task']
      task_template = task_templates[task_name]
      # remove example from template
      if eg_tag in task_template:
        task_template = task_template.split(eg_tag)[0]
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
  
  return pd.DataFrame({'arxiv_id': [str(arxiv_id)]*len(tasks), 'task': tasks, 'excerpt': excerpts, 'blank_templates': templates, 'gt_mapping': gt_mapping, 'annotated_prompts': annotated_prompts, 'annotated': annotation_status})

def make_placeholderdf_for_paper(gdirpath, arxiv_id, task_templates):
  print(f'Arxiv Id: {arxiv_id}  ##############')
  paper_dir = os.path.join(gdirpath, arxiv_id)
  paper_yaml = get_task_yaml_for_paper(arxiv_id, paper_dir)
  taskwise_df = []
  eg_tag = '\n==='
  for elem in paper_yaml: # each task
    if 'task' in elem:
      task_name = elem['task']
      task_template = task_templates[task_name]
      # remove example from template
      if eg_tag in task_template:
        task_template = task_template.split(eg_tag)[0]
      print(f"Task {task_name}")
      # get placeholder GT mapping
      phdict = return_correct_prompt_template_for_task(elem)
      keys = []
      values = []
      baseline_score = []
      for k in phdict:
        keys.append(k)
        values.append(phdict[k])
        if 'score' not in elem['placeholder'][k]:
            print(f'Warning: No score for {k}')
            baseline_score.append('nan')
        else:
            baseline_score.append(elem['placeholder'][k]['score']['Haining'])
      #extend with nph
      Nplh = len(keys)
      excerpt = load_excerpt(paper_dir, elem['source'])
      # placeholder level dataframe for a single task
      taskwise_df.append(pd.DataFrame({'gt_key': keys, 'gt_value': values, 'gpt-4_score': baseline_score, 'arxiv_id': [arxiv_id]*Nplh,  'task': [task_name]*Nplh, 'excerpt': [excerpt]*Nplh, 'blank_template': [task_template]*Nplh}))  
  return pd.concat(taskwise_df)

def retrieve_gpt_answer_from_yaml(df, index, dirpath):
  """Retrieves the GPT answer for comparison."""
  row = df.loc[index]
  arxiv_id = row['arxiv_id']
  paper_dir = os.path.join(dirpath, arxiv_id)
  paper_tasks = get_task_yaml_for_paper(arxiv_id, paper_dir)
  task_id_map = {} #stores task_name -> index in paper_tasks yaml
  for it, elem in enumerate(paper_tasks):
    if 'task' in elem:
      task_id_map[elem['task']]= it
  relevant_task = paper_tasks[task_id_map[row['task']]]
  assert relevant_task['task'] == row['task']
  #placeholder
  ph = row['gt_key']
  if 'LLM' not in relevant_task['placeholder'][ph]:
    gpt_ans = 'NaN'
    print('No LLM Answer')
  else:
    gpt_ans = relevant_task['placeholder'][ph]['LLM']
  print(f"GPT-4 reply for Paper {arxiv_id}, Task {row['task']}, Placeholder '{ph}' = {gpt_ans}")
  return gpt_ans
  
def expand_promptdf_to_placeholderdf(df, index_row, gdirpath):
  gt_dict = json.loads(df.iloc[index_row]['gt_mapping'])
  arxiv_id = str(df.iloc[index_row]['arxiv_id'])
  paper_dir = os.path.join(gdirpath, arxiv_id)
  paper_tasks = get_task_yaml_for_paper(arxiv_id, paper_dir)
  task_id_map = {} #stores task_name -> index in paper_tasks yaml

  for it, elem in enumerate(paper_tasks):
    if 'task' in elem:
      task_id_map[elem['task']]= it

  relevant_task = paper_tasks[task_id_map[df.iloc[index_row]['task']]]
  keys = []
  values = []
  baseline_score = []
  for k in gt_dict:
    keys.append(k)
    values.append(gt_dict[k])
    baseline_score.append(relevant_task['placeholder'][k]['score']['Haining'])
  Nplaceholders = len(keys)
  return pd.DataFrame({'gt_key': keys, 'gt_value': values, 'gpt-4_score': baseline_score, 'arxiv_id': [df.iloc[index_row]['arxiv_id']]*Nplaceholders, 'task': [df.iloc[index_row]['task']]*Nplaceholders, 'excerpt': [df.iloc[index_row]['excerpt']]*Nplaceholders, 'blank_template': [df.iloc[index_row]['blank_templates']]*Nplaceholders})

# Execution
def load_executed_prompts(arxiv_id, dirpath):
  fname = os.path.join(dirpath, arxiv_id, f'{arxiv_id}_auto.md')
  paper_tasks = get_task_yaml_for_paper(arxiv_id, os.path.join(dirpath, arxiv_id))

  task_id_map = {} #stores task_name -> index in paper_tasks yaml
  for it, elem in enumerate(paper_tasks):
    if 'task' in elem:
      task_id_map[elem['task']]= it

  exe_dict = {}
  current_task = None  # Track the current task, initialized to None
  lines = open(fname, 'r').readlines()
  lines = [l.strip() for l in lines]
  il = 0
  while il < len(lines):
    line = lines[il]
    # not skipping any comments
    if line.startswith('## '):
      # New task starts
      current_task = line[3:]
      print(f'Task: {current_task} at Line {il}')
      subcount = il+1
      assert lines[subcount]=='**Prompt:**'
      subcount = il+2 # il+2 is where the prompt starts
      while subcount < len(lines) and not lines[subcount]=='**Completion:**':
        subcount+=1
      p_end = subcount # lines[p_end]=**Completion:**

      while subcount < len(lines) and not lines[subcount].startswith('## '):
        subcount+=1
      c_end = subcount
      try:
        score = paper_tasks[task_id_map[current_task]]['score']
      except KeyError as e:
        print(f'No score for {current_task}')
        score = None
      exe_dict[current_task] = {'prompt': ' '.join(lines[il+2:p_end]), 'lm_execution': ' '.join(lines[p_end+1:c_end]), 'score': score} # p_end+1 is where the completion starts
      il = c_end
  return exe_dict

def consolidate_exec_data(exec_dict):
  executed = []
  for paper in exec_dict:
    paper_tasks = exec_dict[paper]
    for task in paper_tasks:
      reparsed = paper_tasks[task]
      reparsed['task'] = task
      reparsed['paper'] = paper
      executed.append(paper_tasks[task])
  return executed

def parse_scoring_task(exec_task):
  prompt = exec_task['prompt']
  lm_execution = exec_task['lm_execution']
  score = exec_task['score']
  if score:
    target = score['final_answer_accuracy']
  else:
    target = 'NA'
  return {'inputs': f"""PROBLEM: {prompt}\n\nSOLUTION: {lm_execution}\n\nSCORE: """, 'targets': target}

