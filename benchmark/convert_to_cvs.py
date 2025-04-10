import yaml
import re
import os

yaml_path = "test_questions.yaml"

# Load the YAML file
with open(yaml_path, 'r') as file:
    questions_data = yaml.safe_load(file)


import csv

with open("questions.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Question", "Answer", "Reasoning"])
    for entry in questions_data:
        writer.writerow([entry["Question"], entry["Answer"], entry["Reasoning"]])

print("Successfully saved questions.csv")