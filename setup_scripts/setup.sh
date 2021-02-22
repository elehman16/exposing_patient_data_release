export setup_output_folder="setup_outputs/"
mkdir -p $setup_output_folder

## Map each patient subject id to randomly sampled first and last name

python setup_scripts/subject_id_to_name.py \
--input-file data/PATIENTS.csv \
--output-file ${setup_output_folder}/SUBJECT_ID_to_NAME.csv

## Each row contain subject id and one of the notes associated with it.

python setup_scripts/subject_id_to_notes_original.py \
--input-file data/NOTEEVENTS.csv \
--output-file ${setup_output_folder}/SUBJECT_ID_to_NOTES_original.csv

## each row contain subject id and one of the icd9 code associated with it.

python setup_scripts/subject_id_to_icd9.py \
--input-file data/DIAGNOSES_ICD.csv \
--output-file ${setup_output_folder}/SUBJECT_ID_to_ICD9.csv \
--output-descriptions-file ${setup_output_folder}/ICD9_Descriptions.csv