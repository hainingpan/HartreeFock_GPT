#!/usr/bin/env python3
import json
import glob
import sys

def update_notebooks():
    # Find all notebooks
    notebooks = glob.glob('**/*.ipynb', recursive=True)
    
    for notebook_path in notebooks:
        try:
            # Load the notebook
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Track if we made changes
            modified = False
            
            # Check each code cell
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    # Check each line in the cell
                    for i, line in enumerate(cell['source']):
                        if "prompt_text=utils_auto.generate_evalution_prompt(rubric='rubrics3.md', image='image3.md', paper=paper)" in line:
                            # Replace the text
                            cell['source'][i] = line.replace(
                                "prompt_text=utils_auto.generate_evalution_prompt(rubric='rubrics3.md', image='image3.md', paper=paper)",
                                "prompt_text=utils_auto.generate_evalution_prompt(rubric='rubrics3.md', image='image3.md', paper=paper, Gap=0, nu='1/3')"
                            )
                            modified = True
            
            # Save the notebook if modified
            if modified:
                with open(notebook_path, 'w') as f:
                    json.dump(notebook, f, indent=1)
                print(f"Updated: {notebook_path}")
            
        except Exception as e:
            print(f"Error processing {notebook_path}: {e}")

if __name__ == "__main__":
    update_notebooks()
