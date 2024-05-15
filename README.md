# Stance Reasoner: A Zero-Shot Stance Detection on Social Media with Explicit Reasoning, LREC-COLING 2024

This repository contains the code for the stance reasoner method described in the paper "Stance Reasoner: A Zero-Shot Stance Detection on Social Media with Explicit Reasoning".

## Installation

To install the required packages, run the following command in the root directory of the repository:

```bash
poetry install --no-root
```

## Usage

### Data Preprocessing

In order to build the twitter datasets, you need to set up the Twitter API credentials. To do so, create a `.env` file in the root directory of the repository and add the following lines:

```bash
TWITTER_API_BEARER_TOKEN=<YOUR_TWITTER_API_BEARER_TOKEN>
```

To prepare the data, run the following command:

```bash
bash scripts/preprocess_data.sh
```

### Evaluation

To run the evaluation, run the following command:

```bash
bash scripts/evaluate.sh
```
