from glob import glob
import itertools
import os
import re

import argparse
import numpy as np
import pandas as pd
from transformers.pipelines import pipeline
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Generate Contrast sets')
    parser.add_argument('--splits', default='train,validation')
    parser.add_argument('--model', default='facebook/bart-large-cnn', choices=['facebook/bart-large-cnn'])
    parser.add_argument('--num_contrasts', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default=-1, type=int)
    parser.add_argument('--mode', default='generate', choices=['generate', 'merge_batches'])

    args = parser.parse_args()

    splits = args.splits.split(',')

    if args.mode == 'merge':
        for split in splits:
            batches = glob(f'data/contrast_sets/{split}_batch_*.csv')
            batch_idxs = [int(re.search(r'\d+', batch_fn).group(0)) for batch_fn in batches]
            print('Collected {len(batch_idxs} files...')
            np.sort(batch_idxs)
            all_dfs = []
            for batch_idx in batch_idxs:
                batch_fn = f'data/contrast_sets/batch_{batch_idx}.csv'
                df = pd.read_csv(batch_fn)
                all_dfs.append(df)
            full_df = pd.concat(all_dfs)
            out_fn = 'data/full_data.csv'
            full_df.to_csv(out_fn, index=False)
        exit(0)

    print(f'Loading {args.model} summarizer...')
    summarizer = pipeline('summarization', model=args.model, tokenizer=args.model, framework='pt', device=args.device)
    if args.model == 'facebook/bart-large-cnn':
        task_params = {
            'early_stopping': True,
            'length_penalty': 2.0,
            'max_length': 142,
            'min_length': 56,
            'no_repeat_ngram_size': 3,
            'num_beams': 4
        }
    else:
        raise Exception(f'Model={args.model} not supported right now')

    task_params['return_text'] = True
    task_params['truncation'] = True

    contrast_summary_dir = 'data/contrast_sets'
    os.makedirs(contrast_summary_dir, exist_ok=True)
    for split in splits:
        print(f'Loading {split} set...')
        df = pd.read_csv(f'data/{split}_contrast.csv')
        print(f'Loaded {len(df)} examples from {split} set...')
        cols = ['source_orig']
        cols.extend([f'source_contrast_{k + 1}' for k in range(args.num_contrasts)])
        records = df[cols].to_dict('records')
        split_n = len(records)
        all_inputs = list(itertools.chain(*[[record[col] for col in cols] for record in records]))
        all_ids = list(itertools.chain(*[record['id'] * len(cols) for record in records]))
        batch_idxs = list(range(0, len(all_inputs), args.batch_size))
        print(f'{len(all_inputs)} total examples for {split} set.  '
              f'Generating in {len(batch_idxs)} batches of size {args.batch_size}')
        for batch_idx, start_idx in tqdm(enumerate(batch_idxs)):
            end_idx = min(len(all_inputs), start_idx + args.batch_size)
            batch_inputs = all_inputs[start_idx:end_idx]
            batch_ids = all_ids[start_idx:end_idx]
            outputs = list(map(lambda x: x['summary_text'], summarizer(batch_inputs, **task_params)))
            out_fn = os.path.join(contrast_summary_dir, f'{split}_{batch_idx}.csv')
            sum_df = pd.DataFrame({'summary': outputs, 'id': batch_ids})
            sum_df.to_csv(out_fn, index=False)
