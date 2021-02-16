import pandas as pd


def sample_first_name(first_name_file, num_samples):
    """Load the file and get a distribution of first names.
    @param first_name_f is the location of the first names.
    """

    df = pd.read_csv(first_name_file, header=None)
    df.columns = ["name", "gender", "count"]
    df = df[(df["count"] > 10)]
    names = df["name"].sample(n=num_samples, random_state=2021, replace=True).apply(str.title)

    return list(names.values)


def sample_last_name(last_name_file, num_samples):
    """Load the file and get a distribution of last names.
    @param last_name_f is the location of the last names.
    @return a function that is able to generate a new last name (i.e. endless generator).
    """
    df = pd.read_csv(last_name_file)
    df = df[~df.name.isna()]
    df = df[(df["count"] > 400)]
    print(num_samples, len(df))
    names = df["name"].sample(n=num_samples, random_state=2021).apply(str.title)

    return list(names.values)


def run(input_file, output_file, first_name_f, last_name_f):
    patients = pd.read_csv(input_file)
    patients = patients[["SUBJECT_ID", "GENDER"]]

    subject_ids = set(patients["SUBJECT_ID"].values)

    # generate both first and last names
    last_names = sample_last_name(last_name_f, len(subject_ids))
    first_names = sample_first_name(first_name_f, len(subject_ids))

    names = list(zip(first_names, last_names))

    print(len(first_names), len(last_names), len(names))
    print("Unique Names", len(set(names)))

    # add to the data and save
    patients["FIRST_NAME"] = first_names
    patients["LAST_NAME"] = last_names

    patients.sort_values(by="SUBJECT_ID").to_csv(output_file, index=False)


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--first-name-f", default="data/yob1950.txt")
parser.add_argument("--last-name-f", default="data/Names_2010Census.csv")

if __name__ == "__main__":
    """
    Usage:
        - python subject_id_to_name.py --input-file PATIENTS.csv --output-file SUBJECT_ID_to_NAME.csv \
            --first-name-f ... --last-name-f ...

    Output Format:
        SUBJECT_ID,FIRST_NAME,LAST_NAME
        249,Eric,Lehman
        250,Sarthak,Jain
    """

    args = parser.parse_args()
    run(args.input_file, args.output_file, args.first_name_f, args.last_name_f)
