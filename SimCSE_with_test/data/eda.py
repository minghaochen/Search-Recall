import pandas as pd
# a = pd.read_csv('test_query.tsv')
with open('test_query.tsv', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip().split("\t")