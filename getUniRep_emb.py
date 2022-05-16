import argparse
from UniRep.unirep import babbler1900 as babbler
from utils.utils import standardize_dir
import os
import pandas as pd
import numpy as np


# Where model weights are stored.
MODEL_WEIGHT_PATH = "./UniRep/1900_weights"
batch_size = 12
b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)


def parse_dir(dir):
    dir = standardize_dir(dir)
    for subfile in os.listdir(dir):
        curfile = dir + subfile
        if os.path.isdir(curfile):
            parse_dir(curfile)
        else:
            if not curfile.endswith('_seq.csv') and not curfile.endswith('_seqs.csv'):
                continue
            parse_file(curfile)

def parse_file(path):
    df = pd.read_csv(path, header=None)
    seqs = df[df.columns[1]].values.tolist()
    final_rep = list()
    for seq in seqs:
        avg_hidden, final_hidden, final_cell = b.get_rep(seq)
        final_rep.append(final_hidden)
    res_df = pd.DataFrame(np.array(final_rep))
    res_df.to_csv(path.replace('.csv', '_1900emb.csv'), index=False, header=False)

parser = argparse.ArgumentParser(description='Get UniRep representation for dir')
parser.add_argument('--data_path', required=True, help='path to the sequence file')

# Input file should be a csv file, without header and each line should be in the following format:
# proteinId,proteinSequence
args = parser.parse_args()

if not os.path.exists(args.data_path):
    print('ERROR!!! Invalid path, please enter correct path to sequence file!')
    exit(1)

if os.path.isdir(args.data_path):
    parse_dir(args.data_path)
else:
    parse_file(args.data_path)