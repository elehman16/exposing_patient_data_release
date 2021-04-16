#!/bin/bash
set -eu

TOK_MODEL=${TOK_MODEL:-"./OriginalBERT/uncased_L-12_H-768_A-12"}

echo "Using ${TOK_MODEL} for tokenization"

# Preprocessing of text files into BERT format (512 sequence length)
python ./bert/create_pretraining_data.py \
 --do_whole_word_mask=True \
 --input_file=${PRETRAINING_TXT_FILE} \
 --output_file=${PRETRAINING_TXT_FILE}.128.tfrecord \
 --vocab_file=${TOK_MODEL}/vocab.txt \
 --do_lower_case=True \
 --max_seq_length=128  \
 --max_predictions_per_seq=20 \
 --masked_lm_prob=0.15 \
 --random_seed=12345 \
 --dupe_factor=3

# Preprocessing of text files into BERT format (512 sequence length)
python ./bert/create_pretraining_data.py  \
 --do_whole_word_mask=True \
 --input_file=${PRETRAINING_TXT_FILE} \
 --output_file=${PRETRAINING_TXT_FILE}.512.tfrecord \
 --vocab_file=${TOK_MODEL}/vocab.txt \
 --do_lower_case=True \
 --max_seq_length=512 \
 --max_predictions_per_seq=76 \
 --masked_lm_prob=0.15 \
 --random_seed=12345 \
 --dupe_factor=3