#!/usr/bin/env python3
import json
import glob
import re

# Pattern to find (with possible backslashes)
old_pattern = r"\(rubric"
new_text = "(rubric"

# Find all notebooks
notebooks = glob.glob('**/*.ipynb', recursive=True)
count = 0

for notebook_path in notebooks:
    try:
        # Load the notebook
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Track if we made changes
        modified = False
        
        # Process each cell
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # Join all source lines to handle multi-line matches
                source = ''.join(cell.get('source', []))
                
                # Check if our pattern is in the source
                if re.search(old_pattern, source):
                    # Use regex to replace with proper escaping
                    new_source = re.sub(
                        old_pattern, 
                        new_text.replace('(', '\\(').replace(')', '\\)'), 
                        source
                    )
                    
                    # Debug: print what we're changing
                    print(f"Found match in {notebook_path}")
                    print(f"OLD: {source}")
                    print(f"NEW: {new_source}")
                    
                    # Update the cell source
                    cell['source'] = [new_source]
                    modified = True
        
        # Save the notebook if modified
        if modified:
            with open(notebook_path, 'w') as f:
                json.dump(notebook, f, indent=1)
            print(f"Updated: {notebook_path}")
            count += 1
        
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")

print(f"Total notebooks updated: {count}")
