import argparse
import nltk
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
import nltk


def get_rouge_metrics(preds, refs):
    """Computes the rouge metrics.

    Args:
        preds: list. The model predictions.
        refs: list. The references.

    Returns:
        The rouge metrics.
    """
    rouge_output = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {
        "r1": round(rouge_output["rouge1"].mid.fmeasure, 4),
        "r2": round(rouge_output["rouge2"].mid.fmeasure, 4),
        "rL": round(rouge_output["rougeL"].mid.fmeasure, 4)
    }


def get_bertscore_metrics(preds, refs):
    """Computes the bertscore metric.

    Args:
        preds: list. The model predictions.
        refs: list. The references.

    Returns:
        The bertscore metrics.
    """

    bertscore_output = bertscore.compute(predictions=preds, references=refs, lang="en")
    return {
        "p": round(np.mean([v for v in bertscore_output["precision"]]), 4),
        "r": round(np.mean([v for v in bertscore_output["recall"]]), 4),
        "f1": round(np.mean([v for v in bertscore_output["f1"]]), 4)
    }


def get_redundancy_scores(preds):
    sum_unigram_ratio = 0
    sum_bigram_ratio = 0
    sum_trigram_ratio = 0
    all_unigram_ratio = []
    all_bigram_ratio = []
    all_trigram_ratio = []

    sum_redundancy = 0
    stop_words = set(stopwords.words("english"))
    count = CountVectorizer()
    all_redundancy = []

    number_file = len(preds)

    for p in preds:
        all_txt = []
        all_txt.extend(word_tokenize(p.strip()))

        # uniq n-gram ratio
        all_unigram = list(ngrams(all_txt, 1))
        uniq_unigram = set(all_unigram)
        unigram_ratio = len(uniq_unigram) / len(all_unigram)
        sum_unigram_ratio += unigram_ratio

        all_bigram = list(ngrams(all_txt, 2))
        uniq_bigram = set(all_bigram)
        bigram_ratio = len(uniq_bigram) / len(all_bigram)
        sum_bigram_ratio += bigram_ratio

        all_trigram = list(ngrams(all_txt, 3))
        uniq_trigram = set(all_trigram)
        trigram_ratio = len(uniq_trigram) / len(all_trigram)
        sum_trigram_ratio += trigram_ratio

        all_unigram_ratio.append(unigram_ratio)
        all_bigram_ratio.append(bigram_ratio)
        all_trigram_ratio.append(trigram_ratio)

        # NID score
        num_word = len(all_txt)
        new_all_txt = [w for w in all_txt if not w in stop_words]
        new_all_txt = [' '.join(new_all_txt)]

        try:
            x = count.fit_transform(new_all_txt)
            bow = x.toarray()[0]
            max_possible_entropy = np.log(num_word)
            e = entropy(bow)
            redundancy = (1 - e / max_possible_entropy)
            sum_redundancy += redundancy
            all_redundancy.append(redundancy)
        except ValueError:
            continue

    print(f'Number of documents: {number_file}, average unique unigram ratio is {round(sum_unigram_ratio/number_file, 4)}, average unique bigram ratio is {round(sum_bigram_ratio/number_file, 4)}, average unique trigram ratio is {round(sum_trigram_ratio/number_file, 4)}, NID score is {round(sum_redundancy/number_file, 4)}.')
    return all_unigram_ratio, all_bigram_ratio, all_trigram_ratio, all_redundancy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bertscore", default=False, action="store_true", help="If evaluate with bertscore")
    parser.add_argument("--checkpoint", default="allenai/led-base-16384", help="The model checkpoint to use")
    parser.add_argument("--dataset", default="billsum", help="The dataset to use")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("--full", default=False, action="store_true", help="To not use the chunking")
    parser.add_argument("--is_extractive", default=False, action="store_true", help="If used extractive assignment")
    parser.add_argument("--max_input_len", type=int, default=2048, help="The input max size")
    parser.add_argument("--max_output_len", type=int, default=512, help="The output max size")
    parser.add_argument("--n_docs", type=int, default=0, help="The number of training examples")
    parser.add_argument("--prev_loss", default="triplet", help="The loss used in the metric learning task")
    parser.add_argument("--redundancy", default=False, action="store_true", help="If evaluate with redundancy")
    parser.add_argument("--rouge", default=False, action="store_true", help="If evaluate with rouge")
    args = parser.parse_args()

    model_name = "led" if args.checkpoint == "allenai/led-base-16384" else "bart"
    data_dir = "data/"
    predictions_path = "predictions/" + model_name + "_" + args.dataset + "_" + str(args.max_input_len) + "_" + \
                       str(args.max_output_len) + "_" + str(args.epochs) + "_epochs_" + args.prev_loss + "_" + \
                       str(args.is_extractive)
    if args.full:
        predictions_path = predictions_path + "_full"

    if args.n_docs > 0:
        predictions_path += "_" + str(args.n_docs)

    predictions = pd.read_csv(predictions_path)["prediction"].tolist()
    test_dataset = pd.read_csv(data_dir + args.dataset + "_test_set")
    if args.dataset == "pubmed":
        test_dataset = test_dataset.rename(columns={"article": "text", "abstract": "summary"})
    test_dataset = Dataset.from_pandas(test_dataset)

    if args.bertscore:
        bertscore = load_metric("bertscore")
        bertscore_metrics = get_bertscore_metrics(preds=predictions, refs=test_dataset["summary"])
        print(f"\nBERTSCORE-p: {bertscore_metrics['p']}\nBERTSCORE-r: {bertscore_metrics['r']}\nBERTSCORE-f1: {bertscore_metrics['f1']}")

    if args.redundancy:
        nltk.download("stopwords")
        nltk.download("punkt")
        get_redundancy_scores(predictions)
        get_redundancy_scores(test_dataset["text"])
        get_redundancy_scores(test_dataset["summary"])

    if args.rouge:
        rouge = load_metric("rouge")
        rouge_metrics = get_rouge_metrics(preds=predictions, refs=test_dataset["summary"])
        print(f"\nROUGE-1: {rouge_metrics['r1']}\nROUGE-2: {rouge_metrics['r2']}\nROUGE-L: {rouge_metrics['rL']}")


