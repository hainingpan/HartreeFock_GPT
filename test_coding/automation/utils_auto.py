import yaml

import ast


def extract_docstrings(filename):
    """Extract function definitions and their docstrings from a Python file.

    Args:
        filename: Path to the Python file to analyze

    Returns:
        A string containing all function signatures with their docstrings
    """
    with open(filename, "r") as file:
        source = file.read()

    tree = ast.parse(source)
    source_lines = source.split("\n")
    result = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
            # Function has a docstring

            # Get function definition line
            func_line = node.lineno - 1  # Convert to 0-indexed

            # Get docstring node
            docstring_node = node.body[0]
            doc_start_line = docstring_node.lineno - 1

            # Get docstring end line
            doc_end_line = getattr(docstring_node, "end_lineno", None)
            if doc_end_line is not None:
                doc_end_line -= 1  # Convert to 0-indexed
            else:
                # Fallback for Python versions without end_lineno
                doc_end_line = doc_start_line
                for i in range(doc_start_line, len(source_lines)):
                    if i > doc_start_line and (
                        '"""' in source_lines[i] or "'''" in source_lines[i]
                    ):
                        doc_end_line = i
                        break

            # Extract function signature (everything between function line and docstring)
            signature_lines = []
            for i in range(func_line, doc_start_line):
                signature_lines.append(source_lines[i])

            # Extract docstring
            docstring_lines = []
            for i in range(doc_start_line, doc_end_line + 1):
                docstring_lines.append(source_lines[i])

            # Add to result
            result.append("\n".join(signature_lines))
            result.append("\n".join(docstring_lines))
            result.append("")  # Empty line between functions

    return "\n".join(result)


def generate_prompt(
    template, docstring, paper, trial_idx=0, directory="../../", HF="HF.py", save=True
):
    with open(directory + template, "r") as f:
        prompt = f.read()

    with open(directory + docstring, "r") as f:
        docstring = f.read()

    HF_docstring = extract_docstrings('../'+HF)

    with open(directory + "ground_truth.yaml", "r") as f:
        gt = yaml.load(f, Loader=yaml.FullLoader)
    for val in gt:
        if val["arxiv"] == float(paper):
            hamiltonian = val["gt"]
            symmetry = val["symmetry"]
            break
    # prompt = prompt.format(
    #     DOCSTRING="\n" + docstring,
    #     HF_DOCSTRING="\n" + HF_docstring,
    #     HAMILTONIAN=hamiltonian,
    #     SYMMETRY=symmetry,
    # )
    prompt = prompt.replace("{{DOCSTRING}}", '\n'+docstring).replace("{{HF_DOCSTRING}}", '\n'+HF_docstring).replace("{{HAMILTONIAN}}", '\n'+hamiltonian).replace("{{SYMMETRY}}", symmetry)

    output_fn = (
        "prompt_{int}_{decimal}_{trial_idx}".format(
            int=paper.split(".")[0], decimal=paper.split(".")[1], trial_idx=trial_idx
        )
        + ".md"
    )
    if save:
        with open(output_fn, "w") as f:
            f.write(prompt)
    return prompt


### LLM-Generated code
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import anthropic

client = anthropic.Anthropic()


def code_generate(
    prompt,
    client=client,
    model="claude-3-7-sonnet-20250219",
    max_tokens=12800,
    budget_tokens=6400,
    verbose=True,
):
    messages = [{"role": "user", "content": prompt}]

    thinking = {"type": "enabled", "budget_tokens": budget_tokens}
    results = {"thinking": "", "text": ""}
    with client.messages.stream(
        model=model, max_tokens=max_tokens, thinking=thinking, messages=messages
    ) as stream:
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
                        print("#" * 20, "THINKING", "#" * 20, flush=True)
                        print()
                        thinking_flag = False
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    results["text"] += event.delta.text
                    if text_flag and verbose:
                        print("\n")
                        print("#" * 20, "TEXT", "#" * 20, flush=True)
                        print()
                        text_flag = False
                    print(event.delta.text, end="", flush=True)
            elif event.type == "message_stop":
                break
    return results


import re


def extract_code(code_response):
    # get python code. code should be surrounded by ```
    matches = re.findall(r"```(.*?)```", code_response, re.DOTALL)
    proc_codes = []
    for match in matches:
        a = match.split("\n")
        if a[0].lower() == "python":
            code = "\n".join(a[1:])
        else:
            code = "\n".join(a)
        proc_codes.append(code)
    # assume that the actual code is the longest one
    proc_codes = sorted(proc_codes, key=len, reverse=True)
    return proc_codes[0]


def save_code(code, paper, trial_idx):
    output_fn = (
        "code_{int}_{decimal}_{trial_idx}".format(
            int=paper.split(".")[0], decimal=paper.split(".")[1], trial_idx=trial_idx
        )
        + ".py"
    )
    with open(output_fn, "w") as f:
        f.write(code)
        print(f"Code saved to {output_fn}")


### Code evalution
import sys

sys.path.append("../")
import matplotlib.pyplot as plt

import HF
import numpy as np
import inspect


def plot_kspace(kspace):
    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
    ax.scatter(*kspace.T, s=2)
    ax.set_aspect("equal")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    return fig


def plot_matele(mat):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.spy((mat))
    ax.set_title("$H_0$")


def plot_2d_bandstructure(ham, en):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(en.shape[0]):
        ax.plot_trisurf(ham.k_space[:, 0], ham.k_space[:, 1], en[i])
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("E")


def plot_high_symm_bandstructure(k_list, en, ax=None):  # DROP
    if ax is None:
        fig, ax = plt.subplots()
    for e in en:
        k_abs = np.sqrt(np.diff(k_list[:, 0]) ** 2 + np.diff(k_list[:, 1]) ** 2)
        k_abs = np.concatenate([[0], np.cumsum(k_abs)])
        ax.plot(k_abs, e, color="k")
    Nk = (k_list.shape[0] - 1) // 4
    for i in range(5):
        ax.axvline(k_abs[Nk * i], ls="--", color="r")
    ax.set_xticks(
        [k_abs[Nk * i] for i in range(5)], ["$\Gamma$", "K", "M", "$\Gamma$", "K'"]
    )


def plot_2d_false_color_map(ham, en, kmax=6, width=3):
    from matplotlib.ticker import ScalarFormatter

    kmax = min(len(en), kmax)  # number of bands

    fig, ax = plt.subplots(1, kmax, figsize=(3 * kmax, 3), tight_layout=True)
    for idx in range(kmax):
        im = ax[idx].tripcolor(
            ham.k_space[:, 0], ham.k_space[:, 1], en[idx], shading="gouraud"
        )
        axins = ax[idx].inset_axes([1.03, 0., 0.05, 0.99])  
        ax[idx].tricontour(
            ham.k_space[:, 0],
            ham.k_space[:, 1],
            en[idx],
            levels=10,
            colors="k",
            linewidths=1,
        )
        ax[idx].set_title(f"Band {idx+1}")
        ax[idx].set_xlabel("$k_x$")
        ax[idx].set_ylabel("$k_y$")
        
        cbar = plt.colorbar(
            im,
            cax=axins,
            ax=ax[idx],
        )
        cbar.formatter = ScalarFormatter(useOffset=False, useMathText=False)
        cbar.formatter.set_scientific(False)
        cbar.update_ticks()
        ax[idx].set_aspect("equal")

    return fig


def print_gap(ham_int, exp_val, en_int,level=None):
    mean_U = np.abs(ham_int.generate_interacting(exp_val)).mean()
    mean_T = np.abs(ham_int.generate_non_interacting()).mean()
    levels = len(en_int)
    if level is None:
        level = levels // 2
    gap = en_int[level].min() - en_int[level - 1].max()
    print(f"Gap is {gap:.2f}")
    print(f"U/T is {mean_U/mean_T:.2f}")
    print(f"mean_U is {mean_U:.2f}")


### Evaluate code
import base64
import io


def get_gt(
    paper,
    directory="../../",
):
    with open(directory + "ground_truth.yaml", "r") as f:
        gt = yaml.load(f, Loader=yaml.FullLoader)
    for val in gt:
        if val["arxiv"] == float(paper):
            hamiltonian = val["gt"]
            symmetry = val["symmetry"]
            break
    return val


def generate_evalution_prompt(
    rubric,
    image,
    paper,
    prompt_template="evaluation_prompt.md",
    directory="../",
    **kwargs,
):

    with open(directory + prompt_template, "r") as f:
        template = f.read()
    with open(directory + rubric, "r") as f:
        rubric = f.read()
    with open(directory + image, "r") as f:
        image = f.read()
    val = get_gt(paper)
    hamiltonian = f"HAMILTONIAN EQUATION: \n{val['gt']} \nLATTICE: {val['symmetry']}"
    return template.format(
        rubric=rubric.format(lattice=val["symmetry"], **kwargs),
        hamiltonian=hamiltonian,
        image_description=image,
    )


def extract_result_content(string):
    result_tags = re.findall(r"<result>(.*?)</result>", string, re.DOTALL)
    if result_tags:
        result_tags = result_tags[0].strip()
    return result_tags


def vision_eval(
    fig,
    prompt_text,
    budget_tokens=6400,
    max_tokens=12800,
    model="claude-3-7-sonnet-20250219",
    verbose=True,
):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    thinking = {"type": "enabled", "budget_tokens": budget_tokens}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    results = {"thinking": "", "text": ""}
    with client.messages.stream(
        model=model, max_tokens=max_tokens, thinking=thinking, messages=messages
    ) as stream:
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
                        print("#" * 20, "THINKING", "#" * 20, flush=True)
                        print()
                        thinking_flag = False
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    results["text"] += event.delta.text
                    if text_flag and verbose:
                        print("\n")
                        print("#" * 20, "TEXT", "#" * 20, flush=True)
                        print()
                        text_flag = False
                    print(event.delta.text, end="", flush=True)
            elif event.type == "message_stop":
                break
    return results


import os


def save_final_answer(
    paper,
    trial_idx,
    answer1,
    answer2,
    answer3,
    answer4,
    final_answer_file="final_answer.yaml",
):
    # Create the record to be stored for this (paper, trial_idx)
    record_key = f"{trial_idx}"
    record = {
        "paper": paper,
        "trial_idx": trial_idx,
        "answer1": answer1,
        "answer2": answer2,
        "answer3": answer3,
        "answer4": answer4,
    }

    # Load the existing database if it exists, otherwise create an empty dict
    if os.path.exists(final_answer_file):
        with open(final_answer_file, "r") as f:
            try:
                database = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                database = {}
    else:
        database = {}

    # Replace the existing record or add a new one
    database[record_key] = record

    # Save the updated database back to the file
    with open(final_answer_file, "w") as f:
        yaml.safe_dump(database, f)

    print(f"Final answer record for '{record_key}' saved to {final_answer_file}")

def yaml_to_df(yaml_file):
    """Convert YAML file to pandas DataFrame."""
    import pandas as pd
    # Load YAML data from file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Create dataframe from dictionary
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_numeric(df.index)
    df = df.sort_index()
    
    return df