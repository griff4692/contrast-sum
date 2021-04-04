import itertools
import os

import argparse
import pandas as pd
from transformers.pipelines import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Generate Contrast sets')
    parser.add_argument('--splits', default='train,validation')
    parser.add_argument('--model', default='facebook/bart-large-cnn', choices=['facebook/bart-large-cnn'])
    parser.add_argument('--num_contrasts', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default=-1, type=int)

    args = parser.parse_args()

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
        # other alternative if truncation does not work with pipeline
        # model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        # model.generate(
    else:
        raise Exception(f'Model={args.model} not supported right now')

    task_params['return_text'] = True
    task_params['truncation'] = True

    contrast_summary_dir = 'data/contrast_sets'
    os.makedirs(contrast_summary_dir, exist_ok=True)

    splits = args.splits.split(',')
    for split in splits:
        print(f'Loading {split} set...')
        df = pd.read_csv(f'data/{split}_contrast.csv')
        print(f'Loaded {len(df)} examples from {split} set...')
        cols = ['source_orig']
        cols.extend([f'source_contrast_{k + 1}' for k in range(args.num_contrasts)])
        records = df[cols].to_dict('records')
        split_n = len(records)
        all_inputs = list(itertools.chain(*[[record[col] for col in cols] for record in records]))
        batch_idxs = list(range(0, len(all_inputs), args.batch_size))
        print(f'{len(all_inputs)} total examples for {split} set.  '
              f'Generating in {len(batch_idxs)} batches of size {args.batch_size}')
        for batch_idx, start_idx in tqdm(enumerate(batch_idxs)):
            end_idx = min(len(all_inputs), start_idx + args.batch_size)
            batch_inputs = all_inputs[start_idx:end_idx]
            outputs = list(map(lambda x: x['summary_text'], summarizer(batch_inputs, **task_params)))
            out_fn = os.path.join(contrast_summary_dir, f'batch_{batch_idx}.csv')
            sum_df = pd.DataFrame({'summary': outputs})
            sum_df.to_csv(out_fn, index=False)
