#!/bin/bash

Step 1: Copy the template to new notebooks and update trial_idx
for i in {1..2} {4..10}; do
    # Copy the template 
    cp 1106_6060_3.ipynb 1106_6060_${i}.ipynb

    # Replace trial_idx=3 with trial_idx=i
    sed -i "s/trial_idx=3/trial_idx=${i}/g" 1106_6060_${i}.ipynb
    
    echo "Created 1106_6060_${i}.ipynb with trial_idx=${i}"
done

# Step 2: Execute all notebooks with error handling
declare -a failed_notebooks=()
echo "Starting notebook execution..."
for i in {8..9}; do
    echo "Running 1106_6060_${i}.ipynb..."
    if jupyter nbconvert --to notebook --inplace --execute 1106_6060_${i}.ipynb; then
        echo "✓ Successfully executed 1106_6060_${i}.ipynb"
    else
        echo "✗ Error executing 1106_6060_${i}.ipynb"
        failed_notebooks+=($i)
    fi
done

# Print execution summary
echo ""
echo "=== Execution Summary ==="
if [ ${#failed_notebooks[@]} -eq 0 ]; then
  echo "All notebooks executed successfully!"
else
  echo "The following notebooks had execution errors:"
  for nb in "${failed_notebooks[@]}"; do
    echo "- 1106_6060_${nb}.ipynb"
  done
fi