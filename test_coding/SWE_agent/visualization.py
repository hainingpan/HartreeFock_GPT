import os
import re
import argparse

def generate_markdown(base_dir='.', output_file="article_figures.md"):
    """
    Generate a markdown file displaying test figures from article directories with appropriate headings.
    Include text output for test4 if available.
    
    Args:
        base_dir (str): Base directory containing article directories
        output_file (str): Output markdown filename
    """
    # Pattern to match article directories (e.g., 1.1, 2.3, etc.)
    article_pattern = re.compile(r'^\d+\.\d+$')
    
    # Define headings for each test
    headings = {
        "test1.png": "### Check k-points",
        "test2": "### Check noninteracting bandstructure",  # Common heading for 2a and 2b
        "test3.png": "### Check weak coupling limit",
        "test4.png": "### Check strong coupling limit"
    }
    
    # Get all article directories and sort them numerically
    try:
        all_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and article_pattern.match(d)]
        all_dirs.sort(key=lambda x: [float(part) for part in x.split('.')])
    except FileNotFoundError:
        print(f"Error: Base directory '{base_dir}' not found.")
        return
    
    if not all_dirs:
        print(f"Warning: No article directories found in '{base_dir}'.")
        return
    
    # Create the markdown file
    with open(output_file, 'w') as f:
        f.write("# Article Figures\n\n")
        
        for article_dir in all_dirs:
            f.write(f"## Article {article_dir}\n\n")
            
            results_dir = os.path.join(base_dir, article_dir, "hf_env", "results")
            
            if not os.path.exists(results_dir):
                f.write(f"**Note**: Results directory not found for article {article_dir}\n\n")
                continue
            
            # Process test1.png
            process_single_figure(f, results_dir, article_dir, "test1.png", headings["test1.png"])
            
            # Process test2a.png and test2b.png under the same heading
            f.write(headings["test2"] + "\n\n")
            test2a_path = os.path.join(results_dir, "test2a.png")
            test2b_path = os.path.join(results_dir, "test2b.png")
            test2a_relative = os.path.join(article_dir, "hf_env", "results", "test2a.png")
            test2b_relative = os.path.join(article_dir, "hf_env", "results", "test2b.png")
            
            test2_missing = True
            if os.path.exists(test2a_path):
                f.write(f"![test2a.png]({test2a_relative})\n\n")
                test2_missing = False
            else:
                f.write(f"**test2a.png**: Figure missing\n\n")
                
            if os.path.exists(test2b_path):
                f.write(f"![test2b.png]({test2b_relative})\n\n")
                test2_missing = False
            else:
                f.write(f"**test2b.png**: Figure missing\n\n")
                
            if test2_missing:
                f.write("**Note**: All figures for this test are missing\n\n")
            
            # Process test3.png
            process_single_figure(f, results_dir, article_dir, "test3.png", headings["test3.png"])
            
            # Process test4.png with additional output.txt if available
            process_test4(f, results_dir, article_dir, headings["test4.png"])
    print(f"Markdown file '{output_file}' generated successfully.")

def process_single_figure(f, results_dir, article_dir, test_image, heading):
    """Helper function to process a single figure with its heading"""
    image_path = os.path.join(results_dir, test_image)
    relative_path = os.path.join(article_dir, "hf_env", "results", test_image)
    
    f.write(heading + "\n\n")
    
    if os.path.exists(image_path):
        f.write(f"![{test_image}]({relative_path})\n\n")
    else:
        f.write(f"**{test_image}**: Figure missing\n\n")

def process_test4(f, results_dir, article_dir, heading):
    """Process test4 figure with additional output.txt content if available"""
    image_path = os.path.join(results_dir, "test4.png")
    relative_path = os.path.join(article_dir, "hf_env", "results", "test4.png")
    output_txt_path = os.path.join(results_dir, "output.txt")
    
    f.write(heading + "\n\n")
    
    # Check and add the figure
    if os.path.exists(image_path):
        f.write(f"![test4.png]({relative_path})\n\n")
    else:
        f.write(f"**test4.png**: Figure missing\n\n")
    
    # Check and add the output.txt content if it exists
    if os.path.exists(output_txt_path):
        try:
            with open(output_txt_path, 'r') as txt_file:
                output_content = txt_file.read()
                
            f.write("**Gap info:**\n\n")
            f.write("```\n")
            f.write(output_content)
            f.write("\n```\n\n")
        except Exception as e:
            f.write(f"**Note**: Error reading output.txt: {str(e)}\n\n")
    else:
        f.write("**Note**: No output.txt file found for this test\n\n")

def main():
    parser = argparse.ArgumentParser(description='Generate markdown file for article figures with headings.')
    parser.add_argument('--base-dir', default='.', help='Base directory containing article directories')
    parser.add_argument('--output', default='visualization.md', help='Output markdown filename')
    
    args = parser.parse_args()
    generate_markdown(args.base_dir, args.output)

if __name__ == "__main__":
    main()