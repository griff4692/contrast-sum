import itertools
from collections import defaultdict
import regex as re

import pandas as pd
import numpy as np

ENT_OPEN = r'<e id=\d{1,4} type=[A-Z]+>'
ENT_CLOSE = r'<\/e>'
SWAP_PROB = 0.1
HTML_REGEX = r' ?(<[a-z][^>]+>|<\/?[a-z]>) ?'


def swap(arr, a_idx, b_idx):
    prev_a = arr[a_idx]
    arr[a_idx] = arr[b_idx]
    arr[b_idx] = prev_a


def add_noise(source):
    pat = re.compile(HTML_REGEX)
    text_pieces = source.split('||')
    is_tag = list(map(lambda x: pat.search(x) is not None, text_pieces))
    ents_by_type = defaultdict(list)
    dup_types = set()
    for idx, tp in enumerate(text_pieces):
        if tp.startswith('<e id='):
            type = re.search(r'<e id=\d{1,4} type=([_A-Z]+)>', tp).group(1)
            ents_by_type[type].append(idx + 1)
            if len(ents_by_type[type]) > 1:
                dup_types.add(type)

    dup_types = list(dup_types)
    swappable_pairs = []
    for type in dup_types:
        swappable_pairs += itertools.combinations(ents_by_type[type], 2)

    n = len(swappable_pairs)
    rand_vec = np.random.random(size=n)
    to_swap = [swappable_pairs[i] for i in range(n) if rand_vec[i] < SWAP_PROB]
    num_ent_swaps = len(to_swap)

    for swap in to_swap:
        a_idx, b_idx = swap
        prev_a = text_pieces[a_idx]
        text_pieces[a_idx] = text_pieces[b_idx]
        text_pieces[b_idx] = prev_a

    sents = []
    sent_str = ''
    for idx, tp in enumerate(text_pieces):
        if tp == '</s>':
            sents.append(sent_str)
            sent_str = ''
        elif not is_tag[idx]:
            sent_str += tp

    swappable_sents = list(itertools.combinations(range(len(sents)), 2))
    n = len(swappable_sents)
    rand_vec = np.random.random(size=n)
    to_swap = [swappable_sents[i] for i in range(n) if rand_vec[i] < SWAP_PROB]
    num_sent_swaps = len(to_swap)
    for swap in to_swap:
        a_idx, b_idx = swap
        prev_a = sents[a_idx]
        sents[a_idx] = sents[b_idx]
        sents[b_idx] = prev_a
    return ' '.join(sents), num_ent_swaps, num_sent_swaps


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    K = 5
    records = df.to_dict('records')
    for record in records:
        for n in range(K):
            contrast, num_ent_swaps, num_sent_swaps = add_noise(record['source_dec'])
            record[f'source_contrast_{n + 1}'] = contrast
            record[f'source_ent_swaps_{n + 1}'] = num_ent_swaps
            record[f'source_sent_swaps_{n + 1}'] = num_sent_swaps
    contrast_df = pd.DataFrame(records)
    out_fn = 'data/train_contrast.csv'
    contrast_df.to_csv(out_fn, index=False)
