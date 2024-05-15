import os
import time
from pathlib import Path

import datasets
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

_CITATION = """\
@inproceedings{glandt-etal-2021-stance,
    title = "Stance Detection in {COVID}-19 Tweets",
    author = "Glandt, Kyle  and
      Khanal, Sarthak  and
      Li, Yingjie  and
      Caragea, Doina  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.127",
    doi = "10.18653/v1/2021.acl-long.127",
    pages = "1596--1611",
}

"""

_DESCRIPTION = """\
Stance detection on tweets
"""

_HOMEPAGE = "https://github.com/kglandt/stance-detection-in-covid-19-tweets"

_LICENSE = "Apache-2.0 license"

_URL = (
    "https://raw.githubusercontent.com/kglandt/stance-detection-in-covid-19-tweets/main/dataset/"
)

_URLS = {
    "train": [
        _URL + "face_masks_train.csv",
        _URL + "stay_at_home_orders_train.csv",
        _URL + "school_closures_train.csv",
        _URL + "fauci_train.csv",
    ],
    "val": [
        _URL + "face_masks_val.csv",
        _URL + "stay_at_home_orders_val.csv",
        _URL + "school_closures_val.csv",
        _URL + "fauci_val.csv",
    ],
    "test": [
        _URL + "face_masks_test.csv",
        _URL + "stay_at_home_orders_test.csv",
        _URL + "school_closures_test.csv",
        _URL + "fauci_test.csv",
    ],
}


class Covid19Stance(datasets.GeneratorBasedBuilder):
    """Stance detection on tweets."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=VERSION,
            description="Stance detection on tweets",
        )
    ]

    TWEETER_CACHE = Path(os.environ.get("DATA_DIR", os.getcwd() + "/data")) / "twitter_cache"
    assert TWEETER_CACHE.exists(), f"Twitter cache not found at {TWEETER_CACHE}"
    twitter_client = None

    stance_labels = datasets.ClassLabel(names=["against", "favor", "none"])
    sentiment_labels = datasets.ClassLabel(names=["neg", "pos", "other"])
    target_map = {
        "face_masks": "Wearing a Face Mask",
        "stay_at_home_orders": "Stay at Home Orders",
        "school_closures": "Keeping Schools Closed",
        "fauci": "Anthony S. Fauci, M.D.",
    }

    def _info(self):
        features = datasets.Features(
            {
                "tweet_id": datasets.Value("string"),
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
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["val"],
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
        ]

    def get_tweet_by_id(self, tweet_id):
        try:
            CACHE_FILE = self.TWEETER_CACHE / f"{tweet_id}.text"
            # check if tweet is cached
            if CACHE_FILE.exists():
                with CACHE_FILE.open("r") as f:
                    text = f.read()
            else:
                # check if tweeter client is initialized
                if self.twitter_client is None:
                    import tweepy as tw
                    import tweepy.errors as tw_errors

                    barier_token = os.environ.get("TWITTER_API_BEARER_TOKEN")
                    assert barier_token is not None, "TWITTER_API_BEARER_TOKEN is not set"

                    self.twitter_client = tw.Client(bearer_token=self.barier_token)

                tweet = self.twitter_client.get_tweet(tweet_id)
                text = tweet.data.text if tweet.data else ""
                time.sleep(3)

                # cache tweet
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with CACHE_FILE.open("w") as f:
                    f.write(text)

            return text
        except tw_errors.TweepyException as e:
            if e.response.status_code == 429:
                # Rate limit reached, wait for 15 minutes before trying again
                print(f"Rate limit reached. Sleeping for {60} seconds.")
                time.sleep(60)
                return self.get_tweet_by_id(tweet_id)

    def _generate_examples(self, filepath, split):
        # read a list of csv files
        _reader = pd.concat(
            [
                pd.read_csv(
                    f,
                    names=["tweet_id", "target", "label", "opinion towards", "sentiment"],
                    header=0,
                )
                for f in filepath
            ],
            ignore_index=True,
        )

        # get tweet text
        _reader["text"] = _reader["tweet_id"].progress_apply(self.get_tweet_by_id)

        # convert label to int
        _reader["label"] = _reader["label"].str.lower().apply(self.stance_labels.str2int)

        # convert target to title case
        _reader["target"] = _reader["target"].replace(self.target_map)

        # add target
        _reader["split"] = split

        for id, row in _reader.iterrows():
            yield id, row.to_dict()
