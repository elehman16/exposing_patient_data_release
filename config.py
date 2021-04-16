PATIENTS_CSV = "data/PATIENTS.csv"
NOTEEVENTS_CSV = "data/NOTEEVENTS.csv"
DIAGNOSIS_CSV = "data/DIAGNOSES_ICD.csv"

ICD_9_CONVERSION = "data/icd_9_conversion.txt"

BASE_FOLDER = "./"
SUBJECT_ID_to_NAME='setup_outputs/SUBJECT_ID_to_NAME.csv'

MODIFIED_SUBJECT_IDS=f"{BASE_FOLDER}/setup_outputs/reidentified_subject_ids.csv"

SUBJECT_ID_to_ICD9 = f"{BASE_FOLDER}/setup_outputs/SUBJECT_ID_to_ICD9.csv"
SUBJECT_ID_to_Medcat = f"{BASE_FOLDER}/setup_outputs/SUBJECT_ID_to_MedCAT.csv"

condition_type_to_file = {"icd9": SUBJECT_ID_to_ICD9, "medcat": SUBJECT_ID_to_Medcat}

condition_type_to_descriptions = {
    "icd9": f"{BASE_FOLDER}/setup_outputs/ICD9_Descriptions.csv",
    "medcat": f"{BASE_FOLDER}/setup_outputs/MedCAT_Descriptions.csv"
}
