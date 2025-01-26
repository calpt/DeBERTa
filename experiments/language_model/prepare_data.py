# coding: utf-8
from DeBERTa import deberta
import argparse
from tqdm import tqdm
from datasets import load_dataset

def tokenize_data(dataset_name, subset_name, split, output=None, max_seq_length=512):
  p,t=deberta.load_vocab(vocab_path=None, vocab_type='spm', pretrained_id='deberta-v3-base')
  tokenizer=deberta.tokenizers[t](p)
  if output is None:
    output=f'{subset_name}_{split}.spm'
  dataset = load_dataset(dataset_name, subset_name, split=split, streaming=True)
  all_tokens = []
  for l in tqdm(dataset, ncols=80, desc='Loading'):
    l = l["text"]
    if len(l) > 0:
      tokens = tokenizer.tokenize(l)
    else:
      tokens = []
    all_tokens.extend(tokens)

  print(f'Loaded {len(all_tokens)} tokens from {subset_name} {split}')
  lines = 0
  with open(output, 'w', encoding = 'utf-8') as wfs:
    idx = 0
    while idx < len(all_tokens):
      wfs.write(' '.join(all_tokens[idx:idx+max_seq_length-2]) + '\n')
      idx += (max_seq_length - 2)
      lines += 1

  print(f'Saved {lines} lines to {output}')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='The input dataset (HF datasets ID)')
parser.add_argument('-n', '--name', required=True, help='The dataset subset name')
parser.add_argument('-s', '--split', default='train', help='The dataset split')
parser.add_argument('-o', '--output', default=None, help='The output data path')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
args = parser.parse_args()
tokenize_data(args.input, args.name, args.split, args.output, args.max_seq_length)
