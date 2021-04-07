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
    parser.add_argument('--splits', default='validation,train')
    parser.add_argument('--model', default='facebook/bart-large-cnn', choices=['facebook/bart-large-cnn'])
    parser.add_argument('--num_contrasts', default=3, type=int)
    parser.add_argument('--max_n', default=999999999, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--device', default=-1, type=int)
    parser.add_argument('--mode', default='generate', choices=['generate', 'merge'])

    args = parser.parse_args()

    set_size = args.num_contrasts + 1
    assert args.batch_size % set_size == 0

    splits = args.splits.split(',')
    if args.mode == 'merge':
        cols = ['summary_orig']
        cols.extend([f'summary_contrast_{k + 1}' for k in range(args.num_contrasts)])
        for split in splits:
            batches = glob(f'data/sets/{split}_[0-9]*.csv')
            batch_idxs = [int(re.search(r'\d+', batch_fn).group(0)) for batch_fn in batches]
            print(f'Collected {len(batch_idxs)} files...')
            if len(batch_idxs) == 0:
                print('Skipping for now since empty')
                continue
            print(f'Loading {split} set for original source and target')
            source_data = pd.read_csv(f'data/{split}_noise.csv')
            id_to_orig = source_data.set_index('id').to_dict('index')
            np.sort(batch_idxs)
            examples = []
            for i in tqdm(range(len(batch_idxs))):
                batch_fn = f'data/sets/{split}_{batch_idxs[i]}.csv'
                df = pd.read_csv(batch_fn)
                assert len(df) == args.batch_size
                for start_idx in range(0, args.batch_size, set_size):
                    curr_set = df[start_idx:start_idx + set_size].to_dict('records')
                    id = curr_set[0]['id']
                    example = {
                        'id': id,
                        'source': id_to_orig[id]['source_orig'],
                        'target': id_to_orig[id]['target_orig']
                    }
                    for j in range(set_size):
                        example[cols[j]] = curr_set[j]['summary']
                    examples.append(example)
            examples_df = pd.DataFrame(examples)
            out_fn = f'data/{split}_sets.csv'
            print(f'Saving {len(examples_df)} to {out_fn}')
            examples_df.to_csv(out_fn, index=False)
            unique_ids = examples_df['id'].unique().tolist()
            id_sample = set(np.random.choice(unique_ids, size=min(len(unique_ids), 1000), replace=False))
            small_df = examples_df[examples_df['id'].isin(id_sample)]
            small_out_fn = f'data/{split}_sets_small.csv'
            small_df.to_csv(small_out_fn, index=False)
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

    contrast_summary_dir = 'data/sets'
    os.makedirs(contrast_summary_dir, exist_ok=True)
    for split in splits:
        print(f'Loading {split} set...')
        df = pd.read_csv(f'data/{split}_noise.csv')
        prev_n = len(df)
        print(f'Loaded {len(df)} examples from {split} set...')
        df = df[:min(len(df), args.max_n)]
        if len(df) < prev_n:
            print(f'Truncated from {prev_n} to {len(df)}')
        cols = ['source_orig']
        cols.extend([f'source_contrast_{k + 1}' for k in range(args.num_contrasts)])
        records = df[cols + ['id']].to_dict('records')
        split_n = len(records)
        all_inputs = list(itertools.chain(*[[record[col] for col in cols] for record in records]))
        all_ids = list(itertools.chain(*[[record['id']] * len(cols) for record in records]))
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
