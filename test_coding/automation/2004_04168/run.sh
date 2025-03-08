#!/bin/bash

Step 1: Copy the template to new notebooks and update trial_idx
for i in {2..10}; do
    # Copy the template 
    cp 2004_04168_1.ipynb 2004_04168_${i}.ipynb

    sed -i "s/trial_idx=1/trial_idx=${i}/g" 2004_04168_${i}.ipynb
    
    echo "Created 2004_04168_${i}.ipynb with trial_idx=${i}"
done

# Step 2: Execute all notebooks with error handling
declare -a failed_notebooks=()
echo "Starting notebook execution..."
for i in {1..10}; do
    echo "Running 2004_04168_${i}.ipynb..."
    if jupyter nbconvert --to notebook --inplace --execute 2004_04168_${i}.ipynb; then
        echo "✓ Successfully executed 2004_04168_${i}.ipynb"
    else
        echo "✗ Error executing 2004_04168_${i}.ipynb"
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
    echo "- 2004_04168_${nb}.ipynb"
  done
fi