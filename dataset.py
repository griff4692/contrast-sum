import os
import re

import argparse
from nlp.arrow_dataset import Dataset as ArrowDataset
import numpy as np
import pandas as pd
import spacy

from p_tqdm import p_uimap


def process_source(source, spacy_nlp):
    pieces = []
    ent_id = 0
    for sent in spacy_nlp(source).sents:
        pieces.append('<s>')
        curr_idx = sent.start_char
        for ent in sent.ents:
            ent_start, ent_end = ent.start_char, ent.end_char
            prefix = str(source[curr_idx:ent_start])
            if len(prefix) > 0:
                pieces.append(prefix)
            pieces.append(f'<e id={ent_id} type={ent.label_}>')
            pieces.append(str(source[ent_start:ent_end]))
            pieces.append('</e>')
            ent_id += 1
            curr_idx = ent_end
        remaining = str(source[curr_idx:sent.end_char])
        if len(remaining) > 0:
            pieces.append(str(remaining))
        pieces.append('</s>')
    return pieces


def process_target(target_sents, spacy_nlp):
    pieces = []
    ent_id = 0
    for raw_sent in target_sents:
        sent = spacy_nlp(raw_sent)
        pieces.append('<s>')
        curr_idx = 0
        for ent in sent.ents:
            ent_start, ent_end = ent.start_char, ent.end_char
            prefix = str(raw_sent[curr_idx:ent_start])
            if len(prefix) > 0:
                pieces.append(prefix)
            pieces.append(f'<e id={ent_id} type={ent.label_}>')
            pieces.append(str(raw_sent[ent_start:ent_end]))
            pieces.append('</e>')
            ent_id += 1
            curr_idx = ent_end
        remaining = str(raw_sent[curr_idx:])
        if len(remaining) > 0:
            pieces.append(str(remaining))
        pieces.append('</s>')
    return pieces


def decorate(id, source, target, spacy_nlp):
    source = re.sub(r'\|{2,}', ' ', source).replace('&nbsp;', '')
    target = re.sub(r'\|{2,}', ' ', target).replace('&nbsp;', '')
    # source = source.lstrip('(CNN) -- ')
    source_pieces = process_source(source, spacy_nlp)
    target_sents = target.split('\n')
    target_sents = [t.lstrip('NEW: ') for t in target_sents]
    target_pieces = process_target(target_sents, spacy_nlp)
    return {'id': id, 'source_dec': '||'.join(source_pieces), 'target_dec': '||'.join(target_pieces),
            'source_orig': source, 'target_orig': target}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate Contrast Sets for CNN/DM')
    parser.add_argument('--max_n', default=100, type=int)
    parser.add_argument('--spacy', default='en_core_web_sm', choices=[
        'en_core_web_sm',
        'en_core_web_lg',
        'en_core_web_trf'
    ])

    args = parser.parse_args()
    os.makedirs('data', exist_ok=True)
    spacy_nlp = spacy.load(args.spacy)
    splits = ['train', 'validation', 'test']
    data_dir = os.path.expanduser('~/contrast-sum/data/')
    for split in splits:
        split_data = ArrowDataset.from_file(os.path.join(data_dir, f'3.0.0/cnn_dailymail-{split}.arrow'))
        n = len(split_data)
        print(f'Loaded {n} examples for {split} set')
        ids = list(range(len(split_data['article'])))
        sources = split_data['article']
        targets = split_data['highlights']
        data_joined = list(zip(ids, sources, targets))
        if n > args.max_n:
            print(f'Sampling {args.max_n} examples from {n}')
            sample_idxs = np.random.choice(range(n), size=args.max_n, replace=False)
            data_trunc = [data_joined[i] for i in sample_idxs]
            data_joined = data_trunc
        print(f'Process {len(data_joined)} examples for {split} set')
        outputs = list(p_uimap(lambda x: decorate(x[0], x[1], x[2], spacy_nlp=spacy_nlp), data_joined))
        df = pd.DataFrame(outputs)
        out_fn = f'data/{split}.csv'
        print(f'Saving {len(df)} examples to {out_fn}')
        df.to_csv(out_fn, index=False)
