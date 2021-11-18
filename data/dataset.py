import os
import json 
import pandas as pd
import tensorflow as tf

def get_targets(organism):
    targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
    return pd.read_csv(targets_txt, sep='\t')

organism = get_targets("human")


def organism_path(organism):
    return os.path.join('gs://basenji_barnyard/data', organism)


def get_metadata(organism):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)



metadata = get_metadata("human")
