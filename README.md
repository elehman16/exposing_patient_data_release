# Release Version of Does BERT leak Patient Data ?

Some things to keep in mind
---------------------------

* We use the term `reidentified` in many places in the code. This is used to denote if the patient name or their subject id had atleast one occurence in the MIMIC pseudo-reidentified corpus that Clinical BERT was trained. Many experiments are either run only on reidentified patients only (for example, measuring P(condition | patient name)), in other places, we use a method to distinguish whether a patient name appeared in the corpus or not (for example, names_probing.py)

Setup
=====

1. Currently, you need to provide following files in `data` folder. You can get access through `https://mimic.physionet.org` (Note, we are using MIMIC-III dataset).

    - `data/PATIENTS.csv`
    - `data/NOTEEVENTS.csv`
    - `data/DIAGNOSES_ICD.csv`

2. Do initial preprocessing by running `bash setup_scripts/setup.sh`. This will store following information in `setup_outputs/` folder.

    - `SUBJECT_ID, FIRST_NAME, LAST_NAME, GENDER` --- Stored in `SUBJECT_ID_to_NAME.csv`
    - `SUBJECT_ID, TEXT` --- Stored in `SUBJECT_ID_to_NOTES_original.csv`

    - `SUBJECT_ID, CODE` --- Stored in `SUBJECT_ID_to_ICD9.csv`
    - `CODE,DESCRIPTION` --- Stored in `ICD9_Descriptions.csv`

    - `SUBJECT_ID, CODE` --- Stored in `SUBJECT_ID_to_Stanza.csv`
    - `CODE,DESCRIPTION` --- Stored in `Stanza_Descriptions.csv` 


Pseudo-Re-Identification of Names in Notes
=========================================

In this part, we replace known regexes in the patient notes with the actual names we sampled above.
We have two main setups - 

1.a Replace relevant regexes with Names
1.b Replace relevant regexes with Names and Add the Name to beginning of each sentence

Command: `bash setup_scripts/name_insertion.sh`

BERT Training
=============

DO
* `git clone https://github.com/google-research/bert.git`
* `wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip`
* `mkdir -p OriginalBERT/; unzip uncased_L-12_H-768_A-12.zip -d OriginalBERT/`
* `rm uncased_L-12_H-768_A-12.zip`
1. Script that takes in a file of form `SUBJECT_ID_to_NOTES_{type}.csv` and trains BERT. Store BERT model in `ClinicalBERT_{type}/`

Embeddings Training
=============

1. Script that takes in a file of form `SUBJECT_ID_to_NOTES_{type}.csv` and trains Word Embeddings. Store model in `Embedding_{type}/`

Experiments
============

$path_to_model = `ClinicalBERT_{1a|1b}/model_512/` For BERT

$path_to_model = `Embedding_{1a|1b}/{cbow|skipgram}.vectors` For Word Embeddings

## MLM Experiments

### 1. Using MLM, Compute and measure P(condition | name) or P(condition)

```bash
python experiments/MLM/condition_given_name.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9
```

### 2. Using MLM, Compute and measure P(condition | name) - P(condition | name masked)

```bash
python experiments/MLM/condition_given_name_vs_masked.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --metric {probability|rank}
```

### 3. Using MLM, Compute and measure P(name | condition) - P(name | condition masked)

```bash
python experiments/MLM/name_given_condition_vs_masked.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --metric {probability|rank}
```

### 4. Using MLM, Compute and measure P(last name | first name) for reidentified vs unreidentified patients

```bash
python experiments/MLM/first_name_given_last_name.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --metric {probability|rank} --mode {mask_first|mask_last}
```

## Probing Experiments

### 1. Using Probing, Compute and Probe for Score(name, condition). Use common LR/MLP model for all conditions.

```bash
python experiments/probing/all_conditions_probing.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --template-mode {name_and_condition | condition_only} --prober {LR|MLP}
```

### 2. Divide conditions in bins, select 50 conditions in each bin randomly and train individual probers for each condition.

* LR Version

```bash
python experiments/probing/LR_single_conditions_probing.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --frequency-bin {0|1|2|3}
```

* BERT Fine tuned version

```bash
python experiments/probing/FullBERT_single_conditions_probing.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --frequency-bin {0|1|2|3}
```

### 3. Probe for names. Run a probe to distinguish reidentified vs non-reidentified patient names.

```bash
python experiments/probing/names_probing.py --model $path_to_model --tokenizer bert-base-uncased
```

## Cosine Similarity Experiments

### 1. Compute and measure cosine similarity between name and condition wordpiece embeddings from BERT

```bash
python experiments/cosine_similarity/bert_cosine_sim.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9
```

### 2. Compute and measure cosine similarity between name and condition token embeddings under Word Embedding Models (cbow or SG)

```bash
python experiments/cosine_similarity/word_embedding_cosine_sim.py --model-file $path_to_model --condition-type icd9
```