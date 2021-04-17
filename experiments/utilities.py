import config
import pandas as pd
from collections import namedtuple

from typing import Dict, List, Set

import os

def is_debugging_mode() -> bool :
    if "debug_exposing" in os.environ and os.environ["debug_exposing"] == "1":
        print("IN DEBUG MODE")
        return True 

    return False

def get_reidentified_subject_ids_set() -> Set[str]:
    """Return the set of subject ids for patients that had their names occur in notes"""
    subject_ids = pd.read_csv(config.MODIFIED_SUBJECT_IDS).SUBJECT_ID.values
    if is_debugging_mode() :
        subject_ids = subject_ids[:1000]

    return set(subject_ids)


PatientInfo = namedtuple("PatientInfo", field_names=["FIRST_NAME", "LAST_NAME", "GENDER", "CONDITIONS"])

def get_subject_id_to_patient_info(condition_type: str) -> Dict[str, PatientInfo]:
    """Return a Dict mapping [reidentified] subject id to Patient Info associated with it.

    ### Args:
        condition_type: What conditions to include in PatientInfo. Takes value in [icd9, stanza]
    """
    subject_id_to_names = pd.read_csv(config.SUBJECT_ID_to_NAME)
    subject_id_to_names.fillna("", inplace=True)

    assert (
        condition_type in config.condition_type_to_file
    ), f"Unknown Condition type, Select From {list(config.condition_type_to_file.keys())}"

    ## Select ICD9 or Stanza on basis of condition_type
    subject_id_to_condition_codes = pd.read_csv(config.condition_type_to_file[condition_type])

    ## Filter to reidentified subject ids only
    reidentified_subject_ids = get_reidentified_subject_ids_set()
    subject_id_to_names = subject_id_to_names[subject_id_to_names.SUBJECT_ID.isin(reidentified_subject_ids)]
    subject_id_to_condition_codes = subject_id_to_condition_codes[
        subject_id_to_condition_codes.SUBJECT_ID.isin(reidentified_subject_ids)
    ]

    ## Convert to subject_id to list of conditions
    subject_id_to_condition_codes = (
        subject_id_to_condition_codes.groupby("SUBJECT_ID")["CODE"]
        .apply(lambda x: sorted(list(x.values)))
        .reset_index()
    )

    ## Merge condition info with demographics
    subject_id_info = subject_id_to_names.merge(subject_id_to_condition_codes, on="SUBJECT_ID")

    ## Create dictionary mapping subject id to patient info
    first_names = list(subject_id_info["FIRST_NAME"].values)
    last_names = list(subject_id_info["LAST_NAME"].values)
    genders = list(subject_id_info["GENDER"].values)
    conditions = list(subject_id_info["CODE"].values)

    patient_info = [PatientInfo(*tup) for tup in zip(first_names, last_names, genders, conditions)]
    subject_ids = list(subject_id_info["SUBJECT_ID"].values)

    return dict(zip(subject_ids, patient_info))


def get_patient_name_to_is_reidentified() -> Dict[str, int]:
    """Return a Dict mapping patient full name to label indicating whether the patient was reidentified."""
    subject_id_to_name = pd.read_csv(config.SUBJECT_ID_to_NAME)
    subject_id_to_name.fillna("", inplace=True)

    reidentified_subject_ids: Set[str] = get_reidentified_subject_ids_set()

    subject_id_to_name["MODIFIED"] = subject_id_to_name.SUBJECT_ID.apply(
        lambda x: 1 if x in reidentified_subject_ids else 0
    )
    names: List[str] = list((subject_id_to_name.FIRST_NAME + " " + subject_id_to_name.LAST_NAME).values)
    reidentified: List[int] = list(subject_id_to_name.MODIFIED.values)

    labeled_names: Dict[str, int] = dict(zip(names, reidentified))

    return labeled_names


############################################################################################################


def get_condition_code_to_count(condition_type: str) -> Dict[str, int]:
    """Return a dictionary mapping condition code to count of occurrence
    (= how many [reidentified] patients have that code)

    ### Args:
        condition_type: What conditions to return count of. Takes value in [icd9, stanza]
    """
    reidentified_subject_ids = get_reidentified_subject_ids_set()

    assert (
        condition_type in config.condition_type_to_file
    ), f"Unknown Condition type, Select From {list(config.condition_type_to_file.keys())}"

    subject_id_to_icd = pd.read_csv(config.condition_type_to_file[condition_type])

    ## Filter to reidentified subject ids only
    subject_id_to_icd = subject_id_to_icd[subject_id_to_icd.SUBJECT_ID.isin(reidentified_subject_ids)]
    return subject_id_to_icd.CODE.value_counts().to_dict()


def get_condition_code_to_descriptions(condition_type: str) -> Dict[str, str]:
    """Return a dictionary mapping condition code to it's text description

    ### Args:
        condition_type: What conditions to return descriptions of. Takes value in [icd9, stanza]
    """
    code_description_df = pd.read_csv(config.condition_type_to_descriptions[condition_type])
    return dict(
        zip(list(code_description_df["CODE"].values), list(code_description_df["DESCRIPTION"].values))
    )


def filter_condition_code_by_count(
    condition_code_to_count: Dict[str, int], min_count: int, max_count: int
) -> List[str]:
    """Return Sorted List of conditions (by code string) that occur between min_count and max_count

    ### Args:
        condition_code_to_count: Dictionary mapping condition code to count of occurence
        min_count, max_count: int
    """
    conditions = [
        k for k, v in condition_code_to_count.items() if v >= min_count and v <= max_count and len(k) != 1
    ]
    conditions = sorted(list(set(conditions)))
    return conditions


#############################################################################################################


def get_condition_counts_as_vector(
    condition_code_to_count: Dict[str, int], condition_code_to_index: Dict[str, int]
) -> List[int]:
    """Return counts of conditions as list.

    Condition Code is mapped to index in list using condition_code_to_index. This is useful since
    multiple condition code can represent same condition. For example, Stanza.

    ### Args:
        condition_code_to_count: Dict mapping condition code to count
        condition_code_to_index: Dict mapping condition code to index it appears in the list
    """
    num_codes = max(condition_code_to_index.values()) + 1
    label_vector = [0] * num_codes

    for code, count in condition_code_to_count.items():
        if code in condition_code_to_index:
            label_vector[condition_code_to_index[code]] += count

    return label_vector


def get_condition_labels_as_vector(
    condition_code_list: List[str], condition_code_to_index: Dict[str, int]
) -> List[int]:
    """Return list indicating whether a condition code is present in condition code list or not.

    Condition Code is mapped to index in list using condition_code_to_index. This is useful since
    multiple condition code can represent same condition. For example, Stanza.

    ### Args:
        condition_code_list: Set of Conditions that get label 1
        condition_code_to_index: Dict mapping condition code to index it appears in the list
    """
    num_codes = max(condition_code_to_index.values()) + 1
    label_vector = [0] * num_codes

    for code in condition_code_list:
        if code in condition_code_to_index:
            label_vector[condition_code_to_index[code]] = 1

    return label_vector
