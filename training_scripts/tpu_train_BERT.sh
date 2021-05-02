test -d bert || git clone https://github.com/google-research/bert

set -eu

export MODEL_OUTPUT_FOLDER="gs://${bucket_name}/model_outputs/ClinicalBERT_${model_type}"
export PRETRAINING_TXT_FILE="gs://${bucket_name}/model_inputs/${vocab_type}/${notes_type}/notes.sentences*"

export NUM_STEPS=1000000
export NUM_STEPS_512=100000
export INIT_MODEL="gs://${bucket_name}/${init_model}"

# Run pretraining for 128 sequence length
python3 ./bert/run_pretraining.py \
 --init_checkpoint=${INIT_MODEL}/bert_model.ckpt \
 --input_file=${PRETRAINING_TXT_FILE}.128.tfrecord \
 --output_dir=${MODEL_OUTPUT_FOLDER}/model_128/ \
 --do_train=True \
 --do_eval=True \
 --bert_config_file=${INIT_MODEL}/bert_config.json \
 --train_batch_size=128 \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --num_train_steps=${NUM_STEPS} \
 --save_checkpoints_steps=10000 \
 --num_warmup_steps=10000 \
 --learning_rate=2e-5 \
 --use_tpu=True \
 --tpu_name=${TPU_NAME}

gsutil cp ${INIT_MODEL}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_128/bert_config.json
gsutil cp ${INIT_MODEL}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_128/config.json

# Re-name files from model 128 sequence length
gsutil cp ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt-${NUM_STEPS}.index ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt.index
gsutil cp ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt-${NUM_STEPS}.meta  ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt.meta
gsutil cp ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt-${NUM_STEPS}.data-00000-of-00001 ${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt.data-00000-of-00001

# Run pretraining for 512 sequence length
python3 ./bert/run_pretraining.py \
  --input_file=${PRETRAINING_TXT_FILE}.512.tfrecord \
  --output_dir=${MODEL_OUTPUT_FOLDER}/model_512/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=${INIT_MODEL}/bert_config.json \
  --init_checkpoint=${MODEL_OUTPUT_FOLDER}/model_128/model.ckpt \
  --train_batch_size=64 \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --save_checkpoints_steps=10000 \
  --num_train_steps=${NUM_STEPS_512} \
  --num_warmup_steps=5000 \
  --learning_rate=5e-5 \
  --use_tpu=True \
  --tpu_name=${TPU_NAME};

gsutil cp ${INIT_MODEL}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_512/bert_config.json
gsutil cp ${INIT_MODEL}/bert_config.json ${MODEL_OUTPUT_FOLDER}/model_512/config.json

gsutil cp ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt-${NUM_STEPS_512}.index ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt.index
gsutil cp ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt-${NUM_STEPS_512}.meta  ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt.meta
gsutil cp ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt-${NUM_STEPS_512}.data-00000-of-00001 ${MODEL_OUTPUT_FOLDER}/model_512/model.ckpt.data-00000-of-00001