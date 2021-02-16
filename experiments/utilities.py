import random
import config
import pandas as pd

from typing import Dict, List, Set, Tuple

condition_type_to_file = {"icd9": config.SUBJECT_ID_to_ICD9, "stanza": config.SUBJECT_ID_to_Stanza}

from collections import namedtuple

PatientInfo = namedtuple("PatientInfo", field_names=["FIRST_NAME", "LAST_NAME", "GENDER", "CONDITIONS"])


def get_modified_subject_ids_set() -> Set[str]:
    return set(pd.read_csv(config.MODIFIED_SUBJECT_IDS).SUBJECT_ID.values)


def get_subject_id_to_patient_info(condition_type: str) -> Dict[str, PatientInfo]:
    """
    Return a mapping from [modified] SUBJECT_ID to Patient Info associated with it.
    """
    subject_id_to_names = pd.read_csv(config.SUBJECT_ID_to_NAME)
    subject_id_to_names.fillna("", inplace=True)

    assert (
        condition_type in condition_type_to_file
    ), f"Unknown Condition type, Select From {list(condition_type_to_file.keys())}"

    subject_id_to_icd = pd.read_csv(condition_type_to_file[condition_type])

    ## Filter to modified subject ids only
    modified_subject_ids = get_modified_subject_ids_set()
    subject_id_to_names = subject_id_to_names[subject_id_to_names.SUBJECT_ID.isin(modified_subject_ids)]
    subject_id_to_icd = subject_id_to_icd[subject_id_to_icd.SUBJECT_ID.isin(modified_subject_ids)]

    ## Convert to subject_id to list of conditions
    subject_id_to_icd = (
        subject_id_to_icd.groupby("SUBJECT_ID")["CODE"].apply(lambda x: sorted(list(x.values))).reset_index()
    )

    ## Merge condition info with demographics
    subject_id_info = subject_id_to_names.merge(subject_id_to_icd, on="SUBJECT_ID")

    ## Create dictionary mapping subject id to patient info
    first_names = list(subject_id_info["FIRST_NAME"].values)
    last_names = list(subject_id_info["LAST_NAME"].values)
    genders = list(subject_id_info["GENDER"].values)
    conditions = list(subject_id_info["CODE"].values)

    patient_info = [PatientInfo(*tup) for tup in zip(first_names, last_names, genders, conditions)]
    subject_ids = list(subject_id_info["SUBJECT_ID"].values)

    return dict(zip(subject_ids, patient_info))


def get_patient_name_to_is_modified() -> Dict[str, int]:
    """
    Get the patient mappings, where it is a map of patient names to labels.
    @return a mapping of names (str) to integers (0, 1), where 1 is we have
    seen this name, and 0 is we have yet to see this name.
    """
    subject_id_to_name = pd.read_csv(config.SUBJECT_ID_to_NAME)
    subject_id_to_name.fillna("", inplace=True)

    modified_subject_ids: Set[str] = get_modified_subject_ids_set()

    subject_id_to_name["MODIFIED"] = subject_id_to_name.SUBJECT_ID.apply(
        lambda x: 1 if x in modified_subject_ids else 0
    )
    names: List[str] = list((subject_id_to_name.FIRST_NAME + " " + subject_id_to_name.LAST_NAME).values)
    modified: List[int] = list(subject_id_to_name.MODIFIED.values)
    labeled_names: Dict[str, int] = dict(zip(names, modified))

    return labeled_names

############################################################################################################


def get_condition_code_to_count(condition_type: str) -> Dict[str, int]:
    """
    Return a dictionary mapping condition code to count of occurrence (= how many [modified] patients have that code)
    """
    modified_subject_ids = get_modified_subject_ids_set()

    assert (
        condition_type in condition_type_to_file
    ), f"Unknown Condition type, Select From {list(condition_type_to_file.keys())}"

    subject_id_to_icd = pd.read_csv(condition_type_to_file[condition_type])

    ## Filter to modified subject ids only
    subject_id_to_icd = subject_id_to_icd[subject_id_to_icd.SUBJECT_ID.isin(modified_subject_ids)]
    return subject_id_to_icd.CODE.value_counts().to_dict()

def get_condition_code_to_descriptions(condition_type: str) -> Dict[str, str] :
    code_description_df = pd.read_csv(config.condition_type_to_descriptions[condition_type])
    return dict(zip(list(code_description_df["CODE"].values), list(code_description_df["DESCRIPTION"].values)))


def filter_condition_code_by_count(
    condition_code_to_count: Dict[str, int], min_count: int, max_count: int
) -> List[str]:
    """
    Return conditions (ICD or stanza)
    that occur between min_count and max_count times
    and condition name is not of length 1 (chars).
    """
    conditions = [
        k for k, v in condition_code_to_count.items() if v >= min_count and v <= max_count and len(k) != 1
    ]
    conditions = sorted(list(set(conditions)))
    return conditions


#############################################################################################################


def get_condition_counts_as_vector(condition_code_to_count: Dict[str, int], condition_code_to_index: Dict[str, int]) -> List[int]:
    """Given frequency of occurance, and the set of conditions that we will consider,
    produce a baseline prediction based on the frequency.
    @param freq is a mapping of conditions to the number of times it appears.
    @param icd_codes is a list of ICD-9 codes in the given order.
    @return the frequency of conditions given by the order of icd_codes."""
    num_codes = max(condition_code_to_index.values()) + 1
    label_vector = [0] * num_codes

    for code, count in condition_code_to_count.items():
        if code in condition_code_to_index:
            label_vector[condition_code_to_index[code]] += count

    return label_vector


def get_condition_labels_as_vector(
    condition_code_list: List[str], condition_code_to_index: Dict[str, int]
) -> List[int]:
    """
    Create the true y vector for this patient.
    @param condition_code_list: this patients list of conditions.
    @param condition_code_to_index: a dictionary that maps conditions to the index in set_to_use.
    This is useful because 'cold' and 'a cold' are two different strings, but
    should have the same index in set_to_use.
    """
    num_codes = max(condition_code_to_index.values()) + 1
    label_vector = [0] * num_codes

    for code in condition_code_list:
        if code in condition_code_to_index:
            label_vector[condition_code_to_index[code]] = 1

    return label_vector
