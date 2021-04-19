import sys
import pandas as pd
import glob

models = glob.glob("tpu_model_outputs/**/model_*", recursive=True)


files = {
    "icd9": "/condition_given_name/icd9_0_500000/results.txt",
    "medcat": "/condition_given_name/medcat_0_500000/results.txt",
}
for model in models:
    for condition, path in files.items():
        f = open(model + path)
        seg = []
        current_length = None
        info = []
        for line in f:
            if len(line.strip()) == 0:
                continue

            if line.startswith("Length"):
                line = line.split()
                assert line[0] == "Length"
                current_length = line[1]
                continue

            line = line.strip().split()
            assert len(line) == 6
            info.append(
                {
                    "model": model.replace("tpu_model_outputs/", "")
                    .replace("/model_512", "")
                    .replace("/model_128", ""),
                    "condition": condition,
                    "length": current_length,
                    "type": line[0],
                    "metric": line[1].replace(":", ""),
                    "mean": float(line[3]),
                }
            )

info = pd.DataFrame(info)
print(info.model.unique())

map_names = {
    "ClinicalBERT_1a": "Regular Base",
    "ClinicalBERT_1a_Large": "Regular Large",
    "ClinicalBERT_1b": "Name Insertion",
    "ClinicalBERT_templates": "Template Only",
    "ClinicalBERT_1a_Longer": "Regular Base++",
    "ClinicalBERT_1a_Large_Longer": "Regular Large++",
    "Pubmed_ClinicalBERT_1a": "Regular PubmedBase++",
}

keys = [
    "Regular Base",
    "Regular Large",
    "Name Insertion",
    "Template Only",
    "Regular Base++",
    "Regular Large++",
    "Regular PubmedBase++",
]

info["model"] = info["model"].apply(lambda x: map_names[x])
print(info.model.unique())

for t in ["condition_only", "model", "baseline"]:
    print(t)
    print("======")

    for condition in ["icd9", "medcat"]:
        cinfo = info[(info.condition == condition) & (info["type"] == t)].sort_values(
            by="model", key=lambda x: x.apply(lambda y: keys.index(y))
        )
        # if t != "condition_only":
        #     cinfo = cinfo[cinfo.metric != "Spearman"]
        cinfo = cinfo.groupby(["model", "metric"])["mean"].agg("mean")
        cinfo = pd.DataFrame(cinfo).reset_index()

        cinfo = pd.pivot_table(cinfo, values="mean", index="model", columns="metric")

        print(condition)
        print("------")

        print(cinfo.to_latex(index=False))
        print("")
