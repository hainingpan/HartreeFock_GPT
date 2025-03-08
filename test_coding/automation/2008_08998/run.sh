#!/bin/bash

Step 1: Copy the template to new notebooks and update trial_idx
for i in {2..10}; do
    # Copy the template 
    cp 2008_08998_1.ipynb 2008_08998_${i}.ipynb

    # Replace trial_idx=3 with trial_idx=i
    sed -i "s/trial_idx=1/trial_idx=${i}/g" 2008_08998_${i}.ipynb
    
    echo "Created 2008_08998_${i}.ipynb with trial_idx=${i}"
done

# Step 2: Execute all notebooks with error handling
declare -a failed_notebooks=()
echo "Starting notebook execution..."
for i in {2..10}; do
    echo "Running 2008_08998_${i}.ipynb..."
    if jupyter nbconvert --to notebook --inplace --execute 2008_08998_${i}.ipynb; then
        echo "✓ Successfully executed 2008_08998_${i}.ipynb"
    else
        echo "✗ Error executing 2008_08998_${i}.ipynb"
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
    echo "- 2008_08998_${nb}.ipynb"
  done
fi