import csv

import datasets

_CITATION = """\
@inproceedings{mohammad-etal-2016-semeval,
    title = "{S}em{E}val-2016 Task 6: Detecting Stance in Tweets",
    author = "Mohammad, Saif  and
      Kiritchenko, Svetlana  and
      Sobhani, Parinaz  and
      Zhu, Xiaodan  and
      Cherry, Colin",
    booktitle = "Proceedings of the 10th International Workshop on Semantic Evaluation ({S}em{E}val-2016)",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S16-1003",
    doi = "10.18653/v1/S16-1003",
    pages = "31--41",
}
"""

_DESCRIPTION = """\
Stance detection on tweets
"""

_HOMEPAGE = "https://alt.qcri.org/semeval2016/task6/"

_LICENSE = "Apache-2.0 license"

_URL = "http://alt.qcri.org/semeval2016/task6/data/uploads/"
_URLS = {
    "all": _URL + "stancedataset.zip",
}


class Semeval2016Task6a(datasets.GeneratorBasedBuilder):
    """Stance detection on tweets."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=VERSION,
            description="Stance detection on tweets",
        )
    ]

    stance_labels = datasets.ClassLabel(names=["against", "favor", "none"])
    sentiment_labels = datasets.ClassLabel(names=["neg", "pos", "other"])

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "target": datasets.Value("string"),
                "opinion towards": datasets.Value("string"),
                "label": self.stance_labels,
                "sentiment": self.sentiment_labels,
                "split": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # supervised_keys=("text","target","label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files["all"] + "/StanceDataset/train.csv",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": downloaded_files["all"] + "/StanceDataset/test.csv",
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as stances:
            stance_reader = csv.DictReader(stances)

            for id, row in enumerate(stance_reader):
                if row["Target"] == "Donald Trump":
                    continue
                yield id, {
                    "text": row["Tweet"].removesuffix("#SemST"),
                    "target": row["Target"],
                    "opinion towards": row["Opinion Towards"],
                    "label": self.stance_labels.str2int(row["Stance"].lower()),
                    "sentiment": self.sentiment_labels.str2int(row["Sentiment"].lower()),
                    "split": split,
                }
