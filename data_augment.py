from src.variable import *

punctuations = ['。', '.', '．', '!', '！', '?', '？']

# Data is annotated as OSBME, where O is outside, S is single, B is beginning, M is middle, E is end
# While BME data are scarce, we want to augment the data.
# The augmentation is done by splitting the sentence into multiple sentences.
# The new sentence will have the same label as the original sentence.

