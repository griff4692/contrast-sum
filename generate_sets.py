import argparse
import pandas as pd
from transformers.pipelines import pipeline
from tqdm import tqdm

from perturb import add_noise


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Generate Contrast sets')
    parser.add_argument('--splits', default='train,validation')
    parser.add_argument('--model', default='bart-large-cnn', choices=['bart-large-cnn'])
    parser.add_argument('--num_contrasts', default=3, choices=[1, 2, 3, 4, 5], type=int)
    parser.add_argument('--sent_swap_p', default=0.1, type=float)
    parser.add_argument('--ent_swap_p', default=0.1, type=float)
    parser.add_argument('--min_sent_swaps', default=1, type=int)
    parser.add_argument('--min_ent_swaps', default=2, type=int)

    args = parser.parse_args()

    summarizer = pipeline('summarization', model=args.model, tokenizer=args.model, framework='pt')

    if args.model == 'bart-large-cnn':
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

    splits = args.splits.split(',')
    for split in splits:
        df = pd.read_csv(f'data/{split}.csv')
        records = df[['source_orig', 'source_dec']].to_dict('records')
        for record in tqdm(records, total=len(records)):
            orig_source = record['source_orig']
            orig_summary = summarizer(orig_source, **task_params)
            record['model_summary'] = orig_summary
            for i in range(args.num_contrasts):
                contrast_source = add_noise(
                    record['source_dec'], ent_swap_p=args.ent_swap_p, sent_swap_p=args.sent_swap_p,
                    min_ent_swaps=args.min_ent_swaps, min_sent_swaps=args.min_sent_swaps
                )
                contrast_summary = summarizer(contrast_source, **task_params)

                record[f'model_contrast_source_{i}'] = contrast_source
                record[f'model_contrast_summary_{i}'] = contrast_summary

        df_augmented = pd.DataFrame(records)
        out_fn = f'data/{split}_w_sets.csv'
        df_augmented.to_csv(out_fn, index=False)
