# Release Version of Does BERT leak Patient Data ?

Table of Contents
=================

* [Installation](#Installation)
* [Data Preprocessing](#Data-Preprocessing)
* [Pseudo-Re-Identification of Names in Notes](#Pseudo-Re-Identification-of-Names-in-Notes)
* [BERT Pre-Training](#Bert-Pre-Training)
* [Embeddings Pre-Training](#Embeddings-Pre-Training)
* [Experiments](#Experiments)

Some things to keep in mind
---------------------------

* We use the term `reidentified` in many places in the code. This is used to denote if the patient name or their subject id had atleast one occurence in the MIMIC pseudo-reidentified corpus that Clinical BERT was trained. Many experiments are either run only on reidentified patients only (for example, measuring P(condition | patient name)), in other places, we use a method to distinguish whether a patient name appeared in the corpus or not (for example, names_probing.py)

* Below whenever you see something like {...|...|...}, etc, this means you have a choice and can choose one.

* Make sure to **`export PYTHONPATH=.`** before running any command. Run any command from the directory containing this README.

Installation 
============

> bash: conda env create -f conda_env.yml\
> bash: python -m spacy download en\
> bash: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz

Data Preprocessing
==================

1. Currently, you need to provide following files in `data` folder. You can get access through `https://mimic.physionet.org` (Note, we are using MIMIC-III dataset).

    - `data/PATIENTS.csv`
    - `data/NOTEEVENTS.csv`
    - `data/DIAGNOSES_ICD.csv`

2. Do initial preprocessing 

> bash: bash setup_scripts/setup.sh

Output: This will store following information in `setup_outputs/` folder.

    - `SUBJECT_ID, FIRST_NAME, LAST_NAME, GENDER` --- Stored in `SUBJECT_ID_to_NAME.csv`
    - `SUBJECT_ID, TEXT` --- Stored in `SUBJECT_ID_to_NOTES_original.csv`

    - `SUBJECT_ID, CODE` --- Stored in `SUBJECT_ID_to_ICD9.csv`
    - `CODE,DESCRIPTION` --- Stored in `ICD9_Descriptions.csv`

    - `SUBJECT_ID, CODE` --- Stored in `SUBJECT_ID_to_MedCAT.csv`
    - `CODE,DESCRIPTION` --- Stored in `MedCAT_Descriptions.csv` 

Using Physionet
---------------

Instead of regenerating file using above, You can now dowload all the files above from this link ! https://physionet.org/content/clinical-bert-mimic-notes/1.0.0/ . Note, you can access this link if you can access MIMIC data on physionet. 

> bash: wget -r -N -c -np --user <your-physionet-username> --ask-password https://physionet.org/files/clinical-bert-mimic-notes/1.0.0/\
> bash: cp -r physionet.org/files/clinical-bert-mimic-notes/1.0.0/setup_outputs .


Pseudo-Re-Identification of Names in Notes
==========================================

In this part, we replace known regexes in the patient notes with the actual names we sampled above.
We have two main setups - 

1.a Replace relevant regexes with Names\
1.b Replace relevant regexes with Names and Add the Name to beginning of each sentence\
1.c Templates Only version <tt>[CLS] [Patient name] is a yo patient with [Condition] [SEP]</tt>

> bash: bash setup_scripts/name_insertion.sh

Output: This will store following info in `setup_folders/` folder.

    - `SUBJECT_ID_to_NOTES_1a.csv`
    - `SUBJECT_ID_to_NOTES_1b.csv`
    - `SUBJECT_ID_to_NOTES_templates.csv`
    - `reidentified_subject_ids.csv` -- This file contains list of all those patient subject ids that were reidentified in 1a. This is the common set we run experiments for irrespective of which model we are using (1a or 1b)

Using Physionet
---------------

The `setup_outputs` folder copied previously should also contain the reidentified notes, so you don't need to do anything else.


BERT Pre-Training
=============

Using Physionet
---------------

Simplest way to reproduce our experiments is to download the models from physionet directly (Note you don't need to rerun the first command if already done so for `setup_outputs`)

> bash: wget -r -N -c -np --user <your-physionet-username> --ask-password https://physionet.org/files/clinical-bert-mimic-notes/1.0.0/\
> bash: cp -r physionet.org/files/clinical-bert-mimic-notes/1.0.0/model_outputs .

This will download all 7 of our BERT models. Each model contains three files - `bert_config.json`, `config.json` and `pytorch_model.bin` , loadable in huggingface BertModels.



<details>
<summary>If instead you want to train you own BERTs, Click to Expand </summary>

Setup
-----

> git clone https://github.com/google-research/bert.git\
> wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip\
> mkdir -p OriginalBERT/; unzip uncased_L-12_H-768_A-12.zip -d OriginalBERT/\
> rm uncased_L-12_H-768_A-12.zip

Create Data
-----------

Now, we need to convert our notes to tfrecords. We use the code from original BERT repo from google. This step is very time intensive (Single machine may take 1-2 days). But it is trivially parallelizable. Therefore, our code can be used in a distributed setting, where we divide our notes into equal chunks and each chunk is processed on a different machine in cluster (we use 50 machines to reduce preprocessing time to <1hr). We, by default, use vocabulary used for bert-base-uncased , but you can use a different model (for example, pubmedbert from microsoft) by setting the path to corresponding vocab.txt file in environment variable TOK_MODEL.

Note, we use a slurm cluster but we are not including the code for it here. We will let the user decide what is the best way to them to distribute each job. The only parameter the script below needs is `n-jobs` (how many jobs you will run) and `job-num` (for every job, set which chunk to use out of n-jobs using 0-indexing)

```bash
for jobnum in {0..49};
do
    python training_scripts/create_BERT_tfrecords.py \
    --input-file setup_outputs/SUBJECT_ID_to_NOTES_{1a|1b|templates}.csv \
    --output-file model_inputs/bert_base_vocab/{1a|1b|templates}/ \
    --distributed \
    --n-jobs 50 \
    --job-num $jobnum 
done
```

The output will be generated in files `model_inputs/bert_base_vocab/{1a|1b|templates}/notes.sentences.{job-num}-{n-jobs}.{128|512}.tfrecord` .

Note, in our case, all our machines wrote to same output storage, so all tfrecords files end up in same location.

Train Model
-----------

We used free tpus available via TFRC to train our models. We provide the same scripts here (On GPUs, it will take a long time to train and is not recommended). Transfer all files to google cloud platform storage buckets. For example, we transfered in 
`gs://{bucket-name}/model_inputs/bert_base_vocab/{1a|1b|templates}/notes.sentences.{job-num}-{n-jobs}.{128|512}.tfrecord` using gsutils 

> bash: gs -m cp -r model_inputs gs://{bucket-name}\
> bash: gs -m cp -r OriginalBERT gs://{bucket-name}

Once transferred, we can now train model using TPUs. We assume user has familiarity with ctpu utility and can startup a TPU. Within the VM, run (for training from bert-base model)

```bash
bucket_name=${bucket_name} \
vocab_type=bert_base_vocab \
notes_type={1a|1b|templates} \
model_type={1a|1b|templates}_base \
init_model={OriginalBERT/uncased_L-12_H-768_A-12} \
bash training_scripts/tpu_train_BERT.sh
```

Finally, transfer the model back to your system and convert to huggingface format with following -

```bash
bucket_name=${bucket_name} \
model_type={1a|1b|templates}_base \
bash training_scripts/tpu_download_BERT.sh
```

### Output:

Will store BERT model (HuggingFace format) in output_folder `model_outputs/ClinicalBERT_{1a|1b|templates}_base/`

</details>


Embeddings Pre-Training
=============

Train Model
-----------

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

* [Fill In the Blank](#MLM-Experiments)
* [Probing](#Probing-Experiments)
* [Cosine Similarity](#Cosine-Similarity-Experiments)
* [Generation Based](#Generation-Experiments)

$path_to_model = `model_outputs/ClinicalBERT_{model_type}/` For BERT\
$path_to_model = `model_outputs/WordEmbedding_{1a|1b}/{cbow|skipgram}.vectors` For Word Embeddings


## MLM Experiments

### 1. Using MLM, Compute correlation between condition labels for a patient and P(condition | patient name)

```bash
python experiments/MLM/condition_given_name.py --model $path_to_model --tokenizer bert-base-uncased \
--condition-type {icd9|medcat} --template-idx {0|1|2|3}
```

Write metrics to `${path_to_model}/condition_given_name/{condition_type}_{template_idx}` . Results are written for all
length bins. \
For three score types (frequency "baseline", P(condition) "condition_only" and P(condition|patient name) "model"). \
And for three metrics (Area under ROC "AUC", Accuracy at 10 "P@K", Spearman Correlation with Frequency baseline "Spearman")

### 2. Using MLM, Compute and measure P(last name | first name) for reidentified vs unreidentified patients

```bash
python experiments/MLM/first_name_given_last_name.py --model $path_to_model --tokenizer bert-base-uncased \
--metric probability --mode {mask_first|mask_last}
```

Write metrics to `${path_to_model}/first_name_given_last_name/{mode}_{metric}`

## Probing Experiments

### 1. Using Probing, Compute and Probe for Score(name, condition). Use common LR/MLP model for all conditions.

```bash
python experiments/probing/all_conditions_probing.py --model $path_to_model --tokenizer bert-base-uncased \
--condition-type {icd9|medcat} --template-mode {name_and_condition|condition_only} --prober {LR|MLP}
```

### 2. Divide conditions in bins, select 50 conditions in each bin randomly and train individual probers for each condition.

* LR Version

```bash
python experiments/probing/LR_single_conditions_probing.py --model $path_to_model --tokenizer bert-base-uncased \
--condition-type {icd9|medcat} --frequency-bin {0|1|2|3}
```

* BERT Fine tuned version

```bash
python experiments/probing/FullBERT_single_conditions_probing.py --model $path_to_model --tokenizer bert-base-uncased \
--condition-type {icd9|medcat} --frequency-bin {0|1|2|3}
```

### 3. Probe for names. Run a probe to distinguish reidentified vs non-reidentified patient names.

```bash
python experiments/probing/names_probing.py --model $path_to_model --tokenizer bert-base-uncased
```

## Cosine Similarity Experiments

### 1. Compute and measure cosine similarity between name and condition wordpiece embeddings from BERT

```bash
python experiments/cosine_similarity/bert_cosine_sim.py --model $path_to_model --tokenizer bert-base-uncased \
--condition-type {icd9|medcat}
```

### 2. Compute and measure cosine similarity between name and condition token embeddings under Word Embedding Models (cbow or SG)

```bash
python experiments/cosine_similarity/word_embedding_cosine_sim.py --model-file $path_to_model \
--condition-type {icd9|medcat}
```

## Generation Experiments

### 1. Generate Samples from BERT

```bash
MODEL_PATH=$path_to_model TOK_PATH=bert-base-uncased OUT_PATH=$path_to_model python experiments/generation/generate_text.py
```

Each run of above command generate 10000 samples in a file `$path_to_model/samples_{a random hex string}.txt` . We run this command in parallel 50 times to generate 500K samples. At the end, we can run `cat *.txt > samples.txt` in `$path_to_model` directory to combine all samples into single file.


