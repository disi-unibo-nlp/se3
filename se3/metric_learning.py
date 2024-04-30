import argparse
import csv
import os
import torch
from sentence_transformers import InputExample, losses, models, SentenceTransformer, util
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from transformers import set_seed
from zipfile import ZipFile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="The batch size")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument("--loss", default="triplet", help="The loss to use")
    parser.add_argument("--model", default="nlpaueb/legal-bert-base-uncased", help="The model to use")
    parser.add_argument("--seed", type=int, default=1234, help="The seed")
    args = parser.parse_args()

    dataset_path = "data/metric_learning"
    model_path = "models/metric_learning"
    if args.model == "nlpaueb/legal-bert-base-uncased":
        output_path = os.path.join(model_path, "legal-bert-base-uncased" + "-" + args.loss)
    elif args.model == "allenai/scibert_scivocab_uncased":
        output_path = os.path.join(model_path, "scibert-scivocab-uncased" + "-" + args.loss)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        filepath = os.path.join(dataset_path, "wikipedia_sections_triplets.zip")
        util.http_get("https://sbert.net/datasets/wikipedia-sections-triplets.zip", filepath)
        with ZipFile(filepath, "r") as z:
            z.extractall(dataset_path)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    language_model = models.Transformer(args.model, max_seq_length=512).to(device)
    pooling_model = models.Pooling(language_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[language_model, pooling_model])

    print("Read training set")
    train_set = []
    with open(os.path.join(dataset_path, "train.csv"), encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        if args.loss == "triplet":
            train_loss = losses.TripletLoss(model=model)
            for row in reader:
                train_set.append(InputExample(texts=[row["Sentence1"], row["Sentence2"], row["Sentence3"]]))
        elif args.loss == "contrastive":
            train_loss = losses.ContrastiveLoss(model=model)
            for row in reader:
                train_set.append(InputExample(texts=[row["Sentence1"], row["Sentence2"]], label=1))
                train_set.append(InputExample(texts=[row["Sentence1"], row["Sentence3"]], label=0))

    print(f"Use the {args.loss} loss")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    model.fit(train_objectives=[(train_loader, train_loss)],
              epochs=args.epochs,
              warmup_steps=0,
              output_path=output_path)

    print("Read test set")
    test_set = []
    with open(os.path.join(dataset_path, "test.csv"), encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            test_set.append(InputExample(texts=[row["Sentence1"], row["Sentence2"], row["Sentence3"]]))

    test_evaluator = TripletEvaluator.from_input_examples(test_set, name="test")
    test_evaluator(model, output_path=output_path)
