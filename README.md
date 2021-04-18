# Release Version of Does BERT leak Patient Data ?

Some things to keep in mind
---------------------------

* We use the term `reidentified` in many places in the code. This is used to denote if the patient name or their subject id had atleast one occurence in the MIMIC pseudo-reidentified corpus that Clinical BERT was trained. Many experiments are either run only on reidentified patients only (for example, measuring P(condition | patient name)), in other places, we use a method to distinguish whether a patient name appeared in the corpus or not (for example, names_probing.py)

* Below whenever you see something like {...|...|...}, etc, this means you have a choice and can choose one.

* Make sure to **`export PYTHONPATH=.`** before running any command. Run any command from the directory containing this README.

* SciSpacy link is as follows: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz

Installation 
============

> conda env create -f conda_env.yml\
> python -m spacy download en\
> pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz

Setup
=====

1. Currently, you need to provide following files in `data` folder. You can get access through `https://mimic.physionet.org` (Note, we are using MIMIC-III dataset).

    - `data/PATIENTS.csv`
    - `data/NOTEEVENTS.csv`
    - `data/DIAGNOSES_ICD.csv`

2. Do initial preprocessing 

> Command: `PYTHONPATH=. bash setup_scripts/setup.sh`. 

Output: This will store following information in `setup_outputs/` folder.

    - `SUBJECT_ID, FIRST_NAME, LAST_NAME, GENDER` --- Stored in `SUBJECT_ID_to_NAME.csv`
    - `SUBJECT_ID, TEXT` --- Stored in `SUBJECT_ID_to_NOTES_original.csv`

    - `SUBJECT_ID, CODE` --- Stored in `SUBJECT_ID_to_ICD9.csv`
    - `CODE,DESCRIPTION` --- Stored in `ICD9_Descriptions.csv`

    - `SUBJECT_ID, CODE` --- Stored in `SUBJECT_ID_to_MedCAT.csv`
    - `CODE,DESCRIPTION` --- Stored in `MedCAT_Descriptions.csv` 


Pseudo-Re-Identification of Names in Notes
=========================================

In this part, we replace known regexes in the patient notes with the actual names we sampled above.
We have two main setups - 

1.a Replace relevant regexes with Names\
1.b Replace relevant regexes with Names and Add the Name to beginning of each sentence\
1.c Templates Only version <tt>[CLS] [Patient name] is a yo patient with [Condition] [SEP]</tt>

> Command: `PYTHONPATH=. bash setup_scripts/name_insertion.sh`

Output: This will store following info in `setup_folders/` folder.

    - `SUBJECT_ID_to_NOTES_1a.csv`
    - `SUBJECT_ID_to_NOTES_1b.csv`
    - `SUBJECT_ID_to_NOTES_templates.csv`
    - `reidentified_subject_ids.csv` -- This file contains list of all those patient subject ids that were reidentified in 1a. This is the common set we run experiments for irrespective of which model we are using (1a or 1b)


BERT Training
=============

DO
* `git clone https://github.com/google-research/bert.git`
* `wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip`
* `mkdir -p OriginalBERT/; unzip uncased_L-12_H-768_A-12.zip -d OriginalBERT/`
* `rm uncased_L-12_H-768_A-12.zip```

### Command: 

```bash
python training_scripts/train_BERT.py \
--input-file setup_outputs/SUBJECT_ID_to_NOTES_{1a|1b|templates}.csv \
--output-dir model_outputs/ClinicalBERT_{1a|1b|templates}/
```

### Output:

Will store BERT model (HuggingFace format) in output_folder (both 128 and 512 length version) in `model_outputs/ClinicalBERT_{1a|1b|templates}/model_{128|512}/`

Embeddings Training
=============

### Command: 

```bash
python training_scripts/train_word_embeddings.py \
--input-file setup_outputs/SUBJECT_ID_to_NOTES_{1a|1b}.csv \
--output-dir model_outputs/WordEmbeddings_{1a|1b}/ \
--embedding-type {cbow|skipgram}
```

### Output:

Will store gensim Word2Vec model in `model_outputs/WordEmbeddings_{1a|1b}/{cbow|skipgram}.vectors`

Experiments
============

$path_to_model = `ClinicalBERT_{1a|1b}/model_512/` For BERT

$path_to_model = `WordEmbedding_{1a|1b}/{cbow|skipgram}.vectors` For Word Embeddings


## MLM Experiments

### 1. Using MLM, Compute and measure P(condition | name) or P(condition)

```bash
python experiments/MLM/condition_given_name.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9
```

### 2. Using MLM, Compute and measure P(condition | name) - P(condition | name masked)

```bash
python experiments/MLM/condition_given_name_vs_mask.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --metric {probability|rank}
```

### 3. Using MLM, Compute and measure P(name | condition) - P(name | condition masked)

```bash
python experiments/MLM/name_given_condition_vs_mask.py --model $path_to_model --tokenizer bert-base-uncased --condition-type icd9 --metric {probability|rank}
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
