export setup_output_folder="setup_outputs/"

## Replace REGEX with name

python setup_scripts/note_name_insertion.py \
--input-file ${setup_output_folder}/SUBJECT_ID_to_NOTES_original.csv \
--input-names ${setup_output_folder}/SUBJECT_ID_to_NAME.csv \
--output-csv ${setup_output_folder}/SUBJECT_ID_to_NOTES_1a.csv

## Replace REGEX with name + insert name at beginning of sentence

python setup_scripts/note_name_insertion.py \
--input-file ${setup_output_folder}/SUBJECT_ID_to_NOTES_original.csv \
--input-names ${setup_output_folder}/SUBJECT_ID_to_NAME.csv \
--output-csv ${setup_output_folder}/SUBJECT_ID_to_NOTES_1b.csv \
--insert-name-at-bos