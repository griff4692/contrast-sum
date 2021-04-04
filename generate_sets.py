import argparse
import pandas as pd
from transformers.pipelines import pipeline
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to Generate Contrast sets')
    parser.add_argument('--splits', default='train,validation')
    parser.add_argument('--model', default='bart-large-cnn', choices=['bart-large-cnn'])

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
            for k in range(args.num_contrasts):
                contrast_source = record[f'source_contrast_{k + 1}']
                contrast_summary = summarizer(contrast_source, **task_params)
                record[f'summary_contrast_{k + 1}'] = contrast_summary

        df_augmented = pd.DataFrame(records)
        out_fn = f'data/{split}_w_sets.csv'
        df_augmented.to_csv(out_fn, index=False)
