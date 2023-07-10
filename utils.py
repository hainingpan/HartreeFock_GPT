

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

def generate_prompt(kwargs,prompt_dict):
    message={
    'role':
    'user',
    'content':
    prompt_dict[kwargs['task']].format(
        **kwargs).strip()
}
    return message

def assembly_message(sys_msg,user_msg,AI_msg):
    messages = [] + sys_msg
    assert len(user_msg)-len(AI_msg)==1, f'# of user message {len(user_msg)} is not compatible with # of AI_message {len(AI_msg)}'
    messages.append(user_msg[0])
    for user, AI in zip(user_msg[1:],AI_msg):
        messages.append(AI)
        messages.append(user)
    return messages