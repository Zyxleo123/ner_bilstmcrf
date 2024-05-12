import os
import pandas as pd
replacements = {
    'O': 0,
    'S-GPE': 1,
    'S-PER': 2,
    'B-ORG': 3,
    'E-ORG': 4,
    'S-ORG': 5,
    'M-ORG': 6,
    'S-LOC': 7,
    'E-GPE': 8,
    'B-GPE': 9,
    'B-LOC': 10,
    'E-LOC': 11,
    'M-LOC': 12,
    'M-GPE': 13,
    'B-PER': 14,
    'E-PER': 15,
    'M-PER': 16
}
input_dir = 'outputs'
output_dir = 'results'
for input_file in os.listdir(input_dir):
    df = pd.read_csv(os.path.join(input_dir, input_file))
    df = df.replace(replacements)
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = 'id'
    df.to_csv(os.path.join(output_dir, input_file.replace('txt', 'csv')), index=True)
