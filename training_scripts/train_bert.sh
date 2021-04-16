#!/bin/bash
set -eu

BERT_UNCASED="./OriginalBERT/uncased_L-12_H-768_A-12"
NUM_STEPS=500000

# Run pretraining for 128 sequence length
python ./bert/run_pretraining.py \
 --init_checkpoint=./PubmedBERT-base/bert_model.ckpt \
 --input_file=./SUBJECT_ID_to_NOTES_1a.csv.sentences.0-40.128.tfrecord \
 --output_dir=./tmp_outputs/model_128/ \
 --do_train=True \
 --do_eval=True \
 --bert_config_file=./PubmedBERT-base/bert_config.json \
 --train_batch_size=8 \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=${NUM_STEPS} \
 --num_warmup_steps=10 \
 --learning_rate=2e-5

#cp ${BERT_UNCASED}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_128/bert_config.json
#cp ${BERT_UNCASED}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_128/config.json

# Re-name files from model 128 sequence length
#cp ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt-${NUM_STEPS}.index ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt.index
#cp ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt-${NUM_STEPS}.meta  ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt.meta
#cp ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt-${NUM_STEPS}.data-00000-of-00001 ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt.data-00000-of-00001

#mkdir -p ${MODEL_OUTPUT_FOLDER}/model_512

# Run pretraining for 512 sequence length
# python ./bert/run_pretraining.py \
#   --input_file=${PRETRAINING_TXT_FILE}.512.tfrecord \
#   --output_dir=${MODEL_OUTPUT_FOLDER}/model_512 \
#   --do_train=True \
#   --do_eval=True \
#   --bert_config_file=${BERT_UNCASED}/bert_config.json \
#   --init_checkpoint=${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt \
#   --train_batch_size=4 \
#   --max_seq_length=512 \
#   --max_predictions_per_seq=76 \
#   --num_train_steps=${NUM_STEPS} \
#   --num_warmup_steps=5000 \
#   --learning_rate=5e-5;

# cp ${BERT_UNCASED}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_512/bert_config.json
# cp ${BERT_UNCASED}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_512/config.json

# cp ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt-${NUM_STEPS}.index ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt.index
# cp ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt-${NUM_STEPS}.meta  ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt.meta
# cp ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt-${NUM_STEPS}.data-00000-of-00001 ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt.data-00000-of-00001

# python -m transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch \
#  --tf_checkpoint_path ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt \
#  --bert_config_file ${MODEL_OUTPUT_FOLDER}/model_128/bert_config.json \
#  --pytorch_dump_path ${MODEL_OUTPUT_FOLDER}/model_128/pytorch_model.bin

# python -m transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch \
#  --tf_checkpoint_path ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt \
#  --bert_config_file ${MODEL_OUTPUT_FOLDER}/model_512/bert_config.json \
#  --pytorch_dump_path ${MODEL_OUTPUT_FOLDER}/model_512/pytorch_model.bin
