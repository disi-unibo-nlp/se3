import argparse
import numpy as np
import os
import pandas as pd
import torch
from datasets import Dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, set_seed
from rouge_score import rouge_scorer
from segmentation import get_rouge1_precision
from tqdm.auto import tqdm


def process_data_to_model_inputs(batch, max_input_len, max_output_len):
    """Prepares the dataset to be process by transformer models.

    Args:
        batch: The batch to process.
        max_input_len: int. The max input size.
        max_output_len: int: The max output size.

    Returns:
        The batch processed.
    """
    inputs = word_tok(batch["text"], padding="max_length", max_length=max_input_len, truncation=True)
    outputs = word_tok(batch["summary"], padding="max_length", max_length=max_output_len, truncation=True)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids
    batch["labels"] = [
        [-100 if token == word_tok.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    return batch


def prepare_chunks_for_training(dataset, dataset_idx):
    """Prepares the dataset for the training phase by deleting the unlabelled chunks.

    Args:
        dataset: DataFrame. The dataset to process.
        dataset_idx: DataFrame. The chunks' indices.

    Returns:
        new_dataset: DataFrame. The new dataset for the training phase.
        new_dataset_idx: DataFrame. The new chunks' indices.
    """
    chunks = dataset["text"].tolist()
    targets = dataset["summary"].tolist()
    chunks_idx = dataset_idx["idx"].tolist()
    new_chunks = []
    new_targets = []
    new_chunks_idx = []
    # Replace NaN values with empty strings.
    targets = [t if t == t else "" for t in targets]
    for i in chunks_idx:
        # Track the chunks with target.
        cont = 0
        for j in range(i):
            if len(targets[j]) > 0:
                new_chunks.append(chunks[j])
                new_targets.append(targets[j])
                cont += 1
        new_chunks_idx.append(cont)
        del chunks[:i]
        del targets[:i]
    new_dataset = pd.DataFrame(zip(new_chunks, new_targets), columns=["text", "summary"])
    new_dataset_idx = pd.DataFrame(new_chunks_idx, columns=["idx"])
    return new_dataset, new_dataset_idx


def generate_predictions(batch):
    """Generates the predictions on the test set with the fine-tuned model.

    Args:
        batch: The batch to process.

    Returns:
        The batch processed to obtain the predictions.
    """
    inputs = word_tok(batch["text"], padding="max_length", max_length=max_input_length, return_tensors="pt",
                      truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if has_global_attn:
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        predicted_summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               max_length=max_output_length, num_beams=2)
    else:
        predicted_summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                               max_length=max_output_length, num_beams=2)
    batch["predicted_summary"] = word_tok.batch_decode(predicted_summary_ids, skip_special_tokens=True)
    return batch


def concatenate_summaries(preds, dataset_idx):
    """Concatenates the predicted chunks' summaries to rebuild the final summary of each document.

    Args:
        preds: list. The predicted chunks' summaries.
        dataset_idx: DataFrame. The number of chunks per document.

    Returns:
        The final summaries.
    """
    final_summaries = []
    # For each chunk.
    for i in dataset_idx:
        # Build its final summary by concatenating the summaries of its chunks.
        summary = ""
        for j in range(i):
            summary += preds[j]
        final_summaries.append(summary)
        # Delete the first "i" predictions to compute the next document.
        del preds[:i]
    return final_summaries


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


def compute_metrics(pred):
    """Computes the rouge metrics for the evaluation phase.

    Args:
        pred: The model prediction.

    Returns:
        The rouge metrics.
    """
    preds_ids = pred.predictions
    preds_str = word_tok.batch_decode(preds_ids, skip_special_tokens=True)
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = word_tok.pad_token_id
    labels_str = word_tok.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = get_rouge_metrics(preds=preds_str, refs=labels_str)
    return {
        "r1": rouge_output["r1"],
        "r2": rouge_output["r2"],
        "rL": rouge_output["rL"],
        "r-1-2-L-sum": round(rouge_output["r1"] + rouge_output["r2"] + rouge_output["rL"], 4)
    }


def add_doc_index(dataframe_chunked, dataframe_chunked_idx):
    dataframe_chunked["doc_index"] = 0
    index = 0
    for i in range(len(dataframe_chunked_idx)):
        num_chunks = dataframe_chunked_idx.iloc[i]["idx"]
        dataframe_chunked.loc[index:index + num_chunks, "doc_index"] = i
        index += num_chunks
    return dataframe_chunked


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="The batch size for training")
    parser.add_argument("--batch_eval", type=int, default=2, help="The batch size for evaluating")
    parser.add_argument("--checkpoint", default="allenai/led-base-16384", help="The model checkpoint to use")
    parser.add_argument("--dataset", default="billsum", help="The dataset to use")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("--full", default=False, action="store_true", help="To not use the chunking")
    parser.add_argument("--grad_acc", type=int, default=1, help="The gradient accumulation")
    parser.add_argument("--is_extractive", default=False, action="store_true", help="If used extractive assignment")
    parser.add_argument("--is_max", default=False, action="store_true", help="If used max cosine similarity")
    parser.add_argument("--is_paragraph", default=False, action="store_true", help="If used paragraph segmentation")
    parser.add_argument("--max_input_len", type=int, default=2048, help="The input max size")
    parser.add_argument("--max_output_len", type=int, default=512, help="The output max size")
    parser.add_argument("--prev_loss", default="triplet", help="The loss used in the metric learning task")
    parser.add_argument("--predict", default=False, action="store_true", help="To enter in prediction mode")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    parser.add_argument("--test", default=False, action="store_true", help="For testing")
    parser.add_argument("--toy", type=int, default=0, help="The number of toy examples")
    parser.add_argument("--toy_eval", type=int, default=0, help="The number of toy examples for evaluation")
    parser.add_argument("--device", default="cuda", help="The device to use")
    args = parser.parse_args()

    set_seed(args.seed)
    has_global_attn = args.checkpoint == "allenai/led-base-16384"
    if args.device == "cuda":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    max_input_length = args.max_input_len
    max_output_length = args.max_output_len

    model_name = "led" if args.checkpoint == "allenai/led-base-16384" else "bart"

    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_length
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    word_tok = AutoTokenizer.from_pretrained(args.checkpoint)
    rouge = load_metric("rouge")

    data_dir = "data/"
    predictions_path = "predictions/" + model_name + "_" + args.dataset + "_" + str(max_input_length) + "_" + \
                       str(max_output_length) + "_" + str(args.epochs) + "_epochs_" + args.prev_loss + "_" + \
                       str(args.is_extractive)
    model_dir = "models/" + model_name + "/" + args.dataset + "/"
    model_path = model_dir + str(max_input_length) + "_" + str(max_output_length) + "_" + str(args.epochs) + \
                 "_epochs_" + args.prev_loss + "_" + str(args.is_extractive)

    if args.is_paragraph:
        predictions_path += "_paragraph"
        model_path += "_paragraph"

    if args.is_max:
        predictions_path += "_max"
        model_path += "_max"

    if args.toy > 0:
        predictions_path += "_" + str(args.toy)
        model_path += "_" + str(args.toy)

    train_dataset = pd.read_csv(data_dir + args.dataset + "_training_set")
    test_dataset = pd.read_csv(data_dir + args.dataset + "_test_set")
    if args.dataset == "pubmed":
        train_dataset = train_dataset.rename(columns={"article": "text", "abstract": "summary"})
        test_dataset = test_dataset.rename(columns={"article": "text", "abstract": "summary"})
    # Convert the datasets in Huggingface's Datasets format.
    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    train_file_chunked_path = os.path.join(data_dir, model_name + "_" + args.dataset + "_training_set_chunked_"
                                           + str(int(max_input_length/2)) + "_" + str(max_input_length) + "_" +
                                           args.prev_loss + "_" + str(args.is_extractive))
    test_file_chunked_path = os.path.join(data_dir + model_name + "_" + args.dataset + "_test_set_chunked_" +
                                          str(int(max_input_length/2)) + "_" + str(max_input_length) + "_" +
                                          args.prev_loss + "_" + str(args.is_extractive))
    train_file_chunked_idx_path = os.path.join(data_dir + model_name + "_" + args.dataset +
                                               "_training_set_chunked_idx_" + str(int(max_input_length/2)) + "_" +
                                               str(max_input_length) + "_" + args.prev_loss + "_" +
                                               str(args.is_extractive))
    test_file_chunked_idx_path = os.path.join(data_dir + model_name + "_" + args.dataset +
                                              "_test_set_chunked_idx_" + str(int(max_input_length/2)) + "_" +
                                              str(max_input_length) + "_" + args.prev_loss + "_" +
                                              str(args.is_extractive))

    if args.is_paragraph:
        train_file_chunked_path += "_paragraph"
        test_file_chunked_path += "_paragraph"
        train_file_chunked_idx_path += "_paragraph"
        test_file_chunked_idx_path += "_paragraph"

    if args.is_max:
        train_file_chunked_path += "_max"
        test_file_chunked_path += "_max"
        train_file_chunked_idx_path += "_max"
        test_file_chunked_idx_path += "_max"

    if args.full:
        # For toy experiments.
        if args.toy > 0:
            train_dataset = train_dataset.select(range(args.toy))
        if args.toy_eval > 0:
            test_dataset = test_dataset.select(range(args.toy_eval))
            # test_dataset = test_dataset.select(range(args.toy))

        model_path = model_path + "_full"
        predictions_path = predictions_path + "_full"

        print(f"\nWe are using the model '{model_name}' for the dataset '{args.dataset}' with input truncation\n")
        print(f"\nInput size: '{max_input_length}' - Output size '{max_output_length}'\n")

    elif os.path.isfile(train_file_chunked_path) and os.path.isfile(test_file_chunked_path) and \
            os.path.isfile(train_file_chunked_idx_path) and os.path.isfile(test_file_chunked_idx_path):

        train_dataset_chunked = pd.read_csv(train_file_chunked_path)
        train_dataset_chunked_idx = pd.read_csv(train_file_chunked_idx_path)
        test_dataset_chunked = pd.read_csv(test_file_chunked_path)
        test_dataset_chunked_idx = pd.read_csv(test_file_chunked_idx_path)

        train_dataset_chunked = add_doc_index(train_dataset_chunked, train_dataset_chunked_idx)
        test_dataset_chunked = add_doc_index(test_dataset_chunked, test_dataset_chunked_idx)

        # Prepare the training set by dropping the unlabelled chunks (i.e., without target)
        train_dataset_chunked = train_dataset_chunked.dropna()

        # Convert the datasets in Huggingface's Datasets format.
        train_dataset_chunked = Dataset.from_pandas(train_dataset_chunked)
        test_dataset_chunked = Dataset.from_pandas(test_dataset_chunked)
        train_dataset_chunked_idx = Dataset.from_pandas(train_dataset_chunked_idx)
        test_dataset_chunked_idx = Dataset.from_pandas(test_dataset_chunked_idx)

        scorer_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        # Content coverage of train:
        alignment_scores = []
        for i in tqdm(range(len(train_dataset_chunked))):
            chunk = train_dataset_chunked["text"][i]
            target = train_dataset_chunked["summary"][i]
            if target is not None and target != '':
                alignment_scores.append(get_rouge1_precision(scorer_rouge, chunk, target))
        print(f"\nROUGE-1 precision: {round(np.mean(alignment_scores) * 100, 2)}")

        # Content coverage of test:
        alignment_scores = []
        for i in tqdm(range(len(test_dataset_chunked))):
            chunk = test_dataset_chunked["text"][i]
            target = test_dataset_chunked["summary"][i]
            if target is not None and target != '':
                alignment_scores.append(get_rouge1_precision(scorer_rouge, chunk, target))
        print(f"\nROUGE-1 precision: {round(np.mean(alignment_scores) * 100, 2)}")

        exit()

        # For toy experiments.
        if args.toy > 0:
            train_dataset_chunked_idx = train_dataset_chunked_idx.select(range(args.toy))
            train_dataset_chunked = train_dataset_chunked.select(range(int(np.sum(train_dataset_chunked_idx["idx"]))))
            test_dataset_chunked_idx = test_dataset_chunked_idx.select(range(args.toy_eval))
            test_dataset_chunked = test_dataset_chunked.select(range(int(np.sum(test_dataset_chunked_idx["idx"]))))
            # test_dataset_chunked_idx = test_dataset_chunked_idx.select(range(args.toy))
            # test_dataset_chunked = test_dataset_chunked.select(range(int(np.sum(test_dataset_chunked_idx["idx"]))))

        print(f"\nWe are using the model '{model_name}' for the dataset '{args.dataset}'\n")
        print(f"\nInput size: '{max_input_length}' - Output size '{max_output_length}'\n")

    if args.predict:
        if args.full:
            result = test_dataset.map(
                generate_predictions,
                batched=True,
                batch_size=args.batch_eval
            )
            predictions = [x.strip() for x in result["predicted_summary"]]
        else:
            result = test_dataset_chunked.map(
                generate_predictions,
                batched=True,
                batch_size=args.batch_eval
            )
            predictions = concatenate_summaries(result["predicted_summary"], test_dataset_chunked_idx["idx"])
            predictions = [x.strip() for x in predictions]

    elif not os.path.isfile(predictions_path) or args.test:
        if os.path.isfile(model_path) and not args.test:
            print("\nLoad trained weights\n")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("\nPrepare the training set...\n")
            dataset_to_train = train_dataset if args.full else train_dataset_chunked
            train_dataset_model_input = dataset_to_train.map(
                lambda x:
                process_data_to_model_inputs(x, max_input_length, max_output_length),
                batched=True,
                batch_size=args.batch,
                remove_columns=["text", "summary"]
            )

            dataset_to_eval = test_dataset if args.full else test_dataset_chunked
            eval_dataset_model_input = dataset_to_train.map(
                lambda x:
                process_data_to_model_inputs(x, max_input_length, max_output_length),
                batched=True,
                batch_size=args.batch_eval,
                remove_columns=["text", "summary"]
            )
            train_dataset_model_input.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "global_attention_mask", "labels"] if has_global_attn else
                ["input_ids", "attention_mask", "labels"]
            )
            eval_dataset_model_input.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "global_attention_mask", "labels"] if has_global_attn else
                ["input_ids", "attention_mask", "labels"]
            )

            training_args = Seq2SeqTrainingArguments(
                output_dir=model_path,
                per_device_train_batch_size=args.batch,
                gradient_accumulation_steps=args.grad_acc,
                num_train_epochs=args.epochs,
                logging_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                seed=args.seed,
                fp16=True,
                run_name=model_name,
                # evaluation_strategy="epoch",
                # fp16_full_eval=True,
                # load_best_model_at_end=True,
                # metric_for_best_model="r-1-2-L-sum",
                # predict_with_generate=True
            )

            trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=word_tok,
                args=training_args,
                train_dataset=train_dataset_model_input,
                # eval_dataset=eval_dataset_model_input,
                # compute_metrics=compute_metrics,
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            print("\nStart Training...\n")
            trainer.train()

        print("\nStart Evaluation...\n")
        if args.full:
            result = test_dataset.map(
                generate_predictions,
                batched=True,
                batch_size=args.batch_eval
            )
            predictions = [x.strip() for x in result["predicted_summary"]]
        else:
            result = test_dataset_chunked.map(
                generate_predictions,
                batched=True,
                batch_size=args.batch_eval
            )
            predictions = concatenate_summaries(result["predicted_summary"], test_dataset_chunked_idx["idx"])
            predictions = [x.strip() for x in predictions]
        pd.DataFrame(data=predictions, columns=["prediction"]).to_csv(predictions_path)
    else:
        print("\nRead predictions already saved...\n")
        predictions = pd.read_csv(predictions_path)["prediction"].tolist()

    if args.toy > 0:
        test_dataset = test_dataset.select(range(args.toy))

    rouge_metrics = get_rouge_metrics(preds=predictions, refs=test_dataset["summary"])
    print(f"\nROUGE-1: {rouge_metrics['r1']}\nROUGE-2: {rouge_metrics['r2']}\nROUGE-L: {rouge_metrics['rL']}")
