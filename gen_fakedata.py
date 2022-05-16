import pandas as pd
import random
import csv

r"""
cope 
    cp data/ebola/hppi_edgelist.csv db/fake/ddi_edgelist.csv
    cp data/ebola/hppi_edgeweight.csv db/fake/ddi_edgeweight.csv
    cp data/ebola/human_features.csv db/fake/drug_features.csv

    cp data/h1n1/hppi_edgelist.csv db/fake/ppi_edgelist.csv
    cp data/h1n1/hppi_edgeweight.csv db/fake/ppi_edgeweight.csv
    cp data/h1n1/human_features.csv db/fake/protein_features.csv
    
random gen
    pos_train_idx.csv
    pos_test_idx.csv
    neg_train_idx.csv
    neg_test_idx.csv
"""
f_prot = pd.read_csv("db/fake/protein_features.csv")
f_drug = pd.read_csv("db/fake/drug_features.csv")

n_prot = len(f_prot) 
n_drug = len(f_drug) 

#gen random pair
n_pair = 400000
pairs = [(random.randrange(0, n_drug), random.randrange(0, n_prot)) for _ in range(n_pair)]

pairs = list(set(pairs))
n_pair = len(pairs)

pos_pairs = pairs[:n_pair//2]
neg_pairs = pairs[n_pair//2:]

data_dic = dict()
n_pos_pairs = len(pos_pairs)
n_neg_pairs = len(neg_pairs)

data_dic["pos_test_idx.csv"] = pos_pairs[:n_pos_pairs//10]
data_dic["pos_train_idx.csv"] = pos_pairs[n_pos_pairs//10:]
data_dic["neg_test_idx.csv"] = pos_pairs[:n_neg_pairs//10]
data_dic["neg_train_idx.csv"] = pos_pairs[n_neg_pairs//10:]

headers = ["drug","protein"]
for name in data_dic:
    rows = data_dic[name]
    with open("db/fake/"+name,'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
