set -eu

export MODEL_OUTPUT_FOLDER="gs://${bucket_name}/model_outputs/ClinicalBERT_${model_type}/model_512"
export local_output_folder="model_outputs/ClinicalBERT_${model_type}"

mkdir -p $local_output_folder

gsutil cp ${MODEL_OUTPUT_FOLDER}/model.ckpt.* $local_output_folder
gsutil cp ${MODEL_OUTPUT_FOLDER}/config.json $local_output_folder
gsutil cp ${MODEL_OUTPUT_FOLDER}/bert_config.json $local_output_folder

python -m transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch \
 --tf_checkpoint_path ${local_output_folder}/model.ckpt \
 --bert_config_file ${local_output_folder}/bert_config.json \
 --pytorch_dump_path ${local_output_folder}/pytorch_model.bin