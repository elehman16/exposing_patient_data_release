PATIENTS_CSV = "data/PATIENTS.csv"
NOTEEVENTS_CSV = "data/NOTEEVENTS.csv"
DIAGNOSIS_CSV = "data/DIAGNOSES_ICD.csv"

ICD_9_CONVERSION = "data/icd_9_conversion.txt"

BASE_FOLDER = "../exposing_patient_data_v2"
# SUBJECT_ID_to_NAME='setup_outputs/SUBJECT_ID_to_NAMES.csv'
SUBJECT_ID_to_NAME = f"{BASE_FOLDER}/setup_outputs/original_SUBJECT_ID_TO_NAME.csv"

# MODIFIED_SUBJECT_IDS='setup_outputs/modified_subject_ids.csv'
MODIFIED_SUBJECT_IDS = f"{BASE_FOLDER}/setup_outputs/original_MOD_SUBJECT_ID.csv"

SUBJECT_ID_to_ICD9 = f"{BASE_FOLDER}/setup_outputs/SUBJECT_ID_to_ICD9.csv"
SUBJECT_ID_to_Stanza = f"{BASE_FOLDER}/setup_outputs/SUBJECT_ID_to_Stanza.csv"

condition_type_to_file = {"icd9": SUBJECT_ID_to_ICD9, "stanza": SUBJECT_ID_to_Stanza}

condition_type_to_descriptions = {
    "icd9": f"{BASE_FOLDER}/setup_outputs/ICD9_Descriptions.csv",
    "stanza": f"{BASE_FOLDER}/setup_outputs/Stanza_Descriptions.csv",
}
