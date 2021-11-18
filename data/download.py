import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import json
import functools
import torch


def get_targets(organism):
    targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
    return pd.read_csv(targets_txt, sep='\t')


def organism_path(organism):
    return os.path.join('gs://basenji_barnyard/data', organism)


def get_dataset(organism, subset, num_threads=8):
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
    return dataset


def get_metadata(organism):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tfrecord_files(organism, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence,
            'target': target}


# Load dataset
human_dataset = get_dataset('human', 'train').batch(1).repeat()
mouse_dataset = get_dataset('mouse', 'train').batch(1).repeat()
n_examples = 10

for name, dataset in [("human", human_dataset), ("mouse", mouse_dataset)]:
    it = iter(dataset)

    sequence = []
    target = []
    for _ in tqdm(range(n_examples)):
        example = next(it)
        sequence.append(torch.from_numpy(example["sequence"].numpy()))
        target.append(torch.from_numpy(example["target"].numpy()))

    sequence = torch.cat(sequence, dim=0)
    target = torch.cat(target, dim=0)

    assert sequence.shape == torch.Size((n_examples, 131072, 4))
    
    out = {"sequence": sequence, "target": target}
    torch.save(out, f"data/example_data_{name}.pt")
