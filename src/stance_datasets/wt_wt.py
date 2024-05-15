import os
import time
from pathlib import Path

import datasets
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

_CITATION = """\
@inproceedings{conforti-etal-2020-will,
    title = "Will-They-Won{'}t-They: A Very Large Dataset for Stance Detection on {T}witter",
    author = "Conforti, Costanza  and
      Berndt, Jakob  and
      Pilehvar, Mohammad Taher  and
      Giannitsarou, Chryssi  and
      Toxvaerd, Flavio  and
      Collier, Nigel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.157",
    doi = "10.18653/v1/2020.acl-main.157",
    pages = "1715--1724"
}

"""

_DESCRIPTION = """\
Stance detection on tweets
"""

_HOMEPAGE = "https://github.com/cambridge-wtwt/acl2020-wtwt-tweets"

_LICENSE = "Apache-2.0 license"

_URLS = {
    "all": "https://raw.githubusercontent.com/cambridge-wtwt/acl2020-wtwt-tweets/master/wtwt_ids.json",
}


class WTWT(datasets.GeneratorBasedBuilder):
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

    stance_labels = datasets.ClassLabel(names=["refute", "support", "comment", "unrelated"])
    ma_operations = {
        "CVS_AET": {"buyer": "CVS Health", "target": "Aetna", "domain": "healthcare"},
        "CI_ESRX": {"buyer": "Cigna", "target": "Express Scripts", "domain": "healthcare"},
        "ANTM_CI": {"buyer": "Anthem", "target": "Cigna", "domain": "healthcare"},
        "AET_HUM": {"buyer": "Aetna", "target": "Humana", "domain": "healthcare"},
        "FOXA_DIS": {"buyer": "Disney", "target": "21st Century Fox", "domain": "entertainment"},
    }

    twitter_client = None

    def _info(self):
        features = datasets.Features(
            {
                "tweet_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "merger": datasets.Value("string"),
                "target": datasets.Value("string"),
                "label": self.stance_labels,
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
                    "filepath": downloaded_files["all"],
                    "split": "train",
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
        _reader = pd.read_json(filepath, orient="records")

        # rename columns
        _reader = _reader.rename(columns={"stance": "label"})

        # get tweet text
        _reader["text"] = _reader["tweet_id"].progress_apply(self.get_tweet_by_id)

        # convert label to int
        _reader["label"] = _reader["label"].apply(self.stance_labels.str2int)

        # add target
        _reader["target"] = _reader["merger"].apply(
            lambda x: "{buyer} wants to buy {target}".format_map(self.ma_operations[x])
            if x in self.ma_operations
            else ""
        )

        for id, row in _reader.iterrows():
            yield id, row.to_dict()
