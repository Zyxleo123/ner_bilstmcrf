import pandas as pd


df = pd.read_csv('input.csv')
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
df = df.replace(replacements)
df.to_csv('output.csv',index=False)