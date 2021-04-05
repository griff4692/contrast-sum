import itertools
from collections import defaultdict
import regex as re

import argparse
import pandas as pd
from p_tqdm import p_uimap
import numpy as np

ENT_OPEN = r'<e id=\d{1,4} type=[A-Z]+>'
ENT_CLOSE = r'<\/e>'
DEFAULT_SWAP_PROB = 0.1
HTML_REGEX = r' ?(<[a-z][^>]+>|<\/?[a-z]>) ?'


def swap_items(arr, to_swap):
    for a_idx, b_idx in to_swap:
        swap_item(arr, a_idx, b_idx)


def swap_item(arr, a_idx, b_idx):
    prev_a = arr[a_idx]
    arr[a_idx] = arr[b_idx]
    arr[b_idx] = prev_a


def sample_swap_idxs(arr, swap_p):
    to_swap = []
    n = len(arr)
    if n < 2:
        return to_swap
    rand_vec = np.random.random(size=n)
    first_idxs = set([arr[i] for i in range(n) if rand_vec[i] <= swap_p / 2])
    second_idxs = set(arr) - first_idxs
    if len(first_idxs) > len(second_idxs):
        all_combos = itertools.combinations(arr, 2)
        seen = set()
        to_swap = []
        for a, b in all_combos:
            if a in seen or b in seen:
                continue
            to_swap.append((a, b))
            seen.add(a)
            seen.add(b)
        return to_swap
    for idx in first_idxs:
        sample_second_idx = np.random.choice(list(second_idxs), size=1)[0]
        to_swap.append((idx, sample_second_idx))
        second_idxs -= {sample_second_idx}
    return to_swap


def extract_sents(text_pieces):
    pat = re.compile(HTML_REGEX)
    is_tag = list(map(lambda x: pat.search(x) is not None, text_pieces))
    sents = []
    sent_str = ''
    for idx, tp in enumerate(text_pieces):
        if tp == '</s>':
            sents.append(sent_str)
            sent_str = ''
        elif not is_tag[idx]:
            sent_str += tp
    return sents


def ensure_min(all_pairs, sampled_pairs, min_n):
    if len(sampled_pairs) < min_n:
        np.random.shuffle(all_pairs)
        return all_pairs[:min(len(all_pairs), min_n)]
    return sampled_pairs


def add_noise(source, sent_swap_p=DEFAULT_SWAP_PROB, ent_swap_p=DEFAULT_SWAP_PROB, min_sent_swaps=1, min_ent_swaps=2):
    text_pieces = source.split('||')
    ents_by_type = defaultdict(list)
    num_ents = 0
    for idx, tp in enumerate(text_pieces):
        if tp.startswith('<e id='):
            type = re.search(r'<e id=\d{1,4} type=([_A-Z]+)>', tp).group(1)
            ents_by_type[type].append(idx + 1)
            num_ents += 1
    sampled_ent_swaps = []
    all_ent_pairs = []
    for type, idxs in ents_by_type.items():
        all_ent_pairs += list(itertools.combinations(idxs, 2))
        type_pairs = sample_swap_idxs(idxs, ent_swap_p)
        sampled_ent_swaps += type_pairs

    ent_swaps = ensure_min(all_ent_pairs, sampled_ent_swaps, min_ent_swaps)
    num_ent_swaps = len(ent_swaps)
    swap_items(text_pieces, ent_swaps)

    sents = extract_sents(text_pieces)
    num_sents = len(sents)
    sent_idxs = list(range(num_sents))
    all_sent_pairs = list(itertools.combinations(sent_idxs, 2))
    sampled_sent_swaps = sample_swap_idxs(sent_idxs, sent_swap_p)
    sent_swaps = ensure_min(all_sent_pairs, sampled_sent_swaps, min_sent_swaps)
    num_sent_swaps = len(sent_swaps)
    swap_items(text_pieces, sent_swaps)
    return ' '.join(sents), num_ent_swaps, num_ents, num_sent_swaps, num_sents


def process(record, args):
    for k in range(args.K):
        contrast, num_ent_swaps, num_ents, num_sent_swaps, num_sents = add_noise(
            record['source_dec'], ent_swap_p=args.ent_swap_p, sent_swap_p=args.sent_swap_p,
            min_ent_swaps=args.min_ent_swaps, min_sent_swaps=args.min_sent_swaps
        )
        record[f'source_contrast_{k + 1}'] = contrast
        record[f'source_ent_swaps_{k + 1}'] = num_ent_swaps
        record[f'source_sent_swaps_{k + 1}'] = num_sent_swaps
        record['source_num_ents'] = num_ents
        record['source_num_sents'] = num_sents
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Generate Contrast sets')
    parser.add_argument('--splits', default='train,validation')
    parser.add_argument('--num_contrasts', default=3, choices=[1, 2, 3, 4, 5], type=int)
    parser.add_argument('--sent_swap_p', default=0.1, type=float)
    parser.add_argument('--ent_swap_p', default=0.4, type=float)
    parser.add_argument('--min_sent_swaps', default=1, type=int)
    parser.add_argument('--min_ent_swaps', default=5, type=int)
    parser.add_argument('--K', default=3, type=int, help='Number of negative examples to sample')

    args = parser.parse_args()

    splits = args.splits.split(',')
    for split in splits:
        print(f'Loading {split} set')
        df = pd.read_csv(f'data/{split}.csv')
        print(f'Loaded {len(df)} examples')
        records = df.to_dict('records')
        contrast_records = list(p_uimap(lambda record: process(record, args), records))
        contrast_df = pd.DataFrame(contrast_records)
        out_fn = f'data/{split}_contrast.csv'
        contrast_df.to_csv(out_fn, index=False)
