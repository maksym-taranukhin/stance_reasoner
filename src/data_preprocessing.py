import argparse
import os
from datasets import load_dataset

DATASETS_BUILDERS = {
    "semeval2016": "semeval_2016_task_6a.py",
    "covid19": "covid_19.py",
    "wtwt": "wt_wt.py",
}

def main(args):

    file_path = os.path.join("src", "stance_datasets", DATASETS_BUILDERS[args.dataset_name])
    ds = load_dataset(path=file_path, trust_remote_code=True)

    ds.save_to_disk("data/preprocessed/" + args.dataset_name)
    print(ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for stance detection")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")

    args = parser.parse_args()
    main(args)
