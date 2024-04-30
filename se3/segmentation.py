import argparse
import numpy as np
import os
import pandas as pd
import pysbd
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed


def self_segmentation(doc, s_bert_model, word_tok, sent_tok, dev, min_len, max_len, is_max):
    """Segments a document by focusing on the entailment meaning of neighboring sentences.

    Args:
        doc: string. The document to process.
        s_bert_model: SentenceTransformer. The model to get the sentences' embeddings.
        word_tok: Tokenizer. The sub-words' tokenizer.
        sent_tok: Segmenter. The sentences' tokenizer.
        dev: device. The device to use for the computation.
        min_len: int. The chunk's min size.
        max_len: int. The chunk's max size.
        is_max: bool. To apply the max cosine similarity.

    Returns:
        A list of chunks.
    """
    chunks = []
    doc = str(doc).lower()
    sentences = sent_tok.segment(doc)
    dictionary = None
    if len(word_tok(doc)["input_ids"]) < max_len:
        chunks.append(sentences)
    else:
        sent_embeddings = s_bert_model.encode(sentences, convert_to_tensor=True, device=dev)
        dictionary = dict(zip(sentences, sent_embeddings))
        count = 0
        curr_chunk = []
        curr_chunk_len = 0
        while count < len(sentences) - 1:
            curr_sent = sentences[count]
            curr_sent_len = len(word_tok(curr_sent)["input_ids"])
            if curr_chunk_len < min_len:
                if (curr_chunk_len + curr_sent_len) > max_len:
                    if curr_chunk_len == 0:
                        curr_chunk.append(curr_sent)
                        curr_chunk_len += curr_sent_len
                    else:
                        chunks.append(curr_chunk)
                        # Create a new chunk.
                        curr_chunk = [curr_sent]
                        curr_chunk_len = curr_sent_len
                else:
                    curr_chunk.append(curr_sent)
                    curr_chunk_len += curr_sent_len
                count += 1
            else:  # The current chunk size is between "min_len" and "max_len", do auto-segmentation.
                if (curr_chunk_len + curr_sent_len) > max_len:
                    chunks.append(curr_chunk)
                    # Create a new chunk.
                    curr_chunk = [curr_sent]
                    curr_chunk_len = curr_sent_len
                    count += 1
                else:
                    # Do the look-ahead to check the similarity with the next chunk.
                    next_chunk = []
                    next_chunk_embeds = []
                    next_chunk_len = 0
                    for s in sentences[count + 1:]:  # For each sentence after the current one.
                        if next_chunk_len < min_len:
                            s_len = len(word_tok(s)["input_ids"])
                            if (next_chunk_len + s_len) > max_len:
                                break  # Stop the look-ahead.
                            else:
                                next_chunk.append(s)
                                next_chunk_embeds.append(dictionary.get(s))
                                next_chunk_len += s_len
                        else:  # The next chunk size is between "min_len" and "max_len", stop the look-ahead.
                            break
                    if next_chunk_len > 0:
                        if (curr_sent_len + next_chunk_len) < max_len:
                            curr_chunk_embeds = torch.stack([dictionary.get(s) for s in curr_chunk])
                            curr_sent_emb = dictionary.get(curr_sent)
                            curr_sim = util.pytorch_cos_sim(curr_sent_emb, curr_chunk_embeds)
                            next_sim = util.pytorch_cos_sim(curr_sent_emb, torch.stack(next_chunk_embeds))

                            final_sim = curr_sim.max() > next_sim.max() if is_max else curr_sim.mean() > next_sim.mean()

                            if final_sim:
                                curr_chunk.append(curr_sent)  # Add the current sentence to the current chunk.
                                curr_chunk_len += curr_sent_len
                                count += 1
                            else:
                                chunks.append(curr_chunk)
                                # Add the current sentence to the next chunk along with the future sentences.
                                curr_chunk = [curr_sent, *next_chunk]
                                curr_chunk_len = np.sum([len(word_tok(s)["input_ids"]) for s in curr_chunk])
                                count += len(curr_chunk)
                        else:
                            curr_chunk.append(curr_sent)
                            curr_chunk_len += curr_sent_len
                            count += 1
                    else:
                        curr_chunk.append(curr_sent)
                        curr_chunk_len += curr_sent_len
                        count += 1
        last_sent = sentences[-1]
        last_sent_len = len(word_tok(last_sent)["input_ids"])
        if (curr_chunk_len + last_sent_len) > max_len:
            chunks.append(curr_chunk)
            # Create a new chunk.
            chunks.append([last_sent])
        else:
            curr_chunk.append(last_sent)
            chunks.append(curr_chunk)
    chunks = [" ".join(c) for c in chunks]
    return chunks, dictionary


def paragraph_segmentation(doc, word_tok, sent_tok, min_len, max_len):
    chunks = []
    doc = str(doc).lower()
    sentences = sent_tok.segment(doc)
    doc_len = len(word_tok(doc)["input_ids"])
    if doc_len < max_len:
        chunks.append(sentences)
    else:
        curr_chunk = []
        curr_chunk_len = 0
        for count in range(len(sentences) - 1):
            curr_sent = sentences[count]
            curr_sent_len = len(word_tok(curr_sent)["input_ids"])
            if curr_chunk_len < min_len:
                if (curr_chunk_len + curr_sent_len) > max_len:
                    if curr_chunk_len == 0:
                        curr_chunk.append(curr_sent)
                        curr_chunk_len += curr_sent_len
                    else:
                        chunks.append(curr_chunk)
                        # Create a new chunk.
                        curr_chunk = [curr_sent]
                        curr_chunk_len = curr_sent_len
                else:
                    curr_chunk.append(curr_sent)
                    curr_chunk_len += curr_sent_len
            else:
                chunks.append(curr_chunk)
                # Create a new chunk.
                curr_chunk = [curr_sent]
                curr_chunk_len = curr_sent_len
        last_sent = sentences[-1]
        last_sent_len = len(word_tok(last_sent)["input_ids"])
        if (curr_chunk_len + last_sent_len) > max_len:
            chunks.append(curr_chunk)
            # Create a new chunk.
            chunks.append([last_sent])
        else:
            curr_chunk.append(last_sent)
            chunks.append(curr_chunk)
    chunks = [" ".join(c) for c in chunks]
    return chunks


def get_rouge1_precision(scorer, chunk, sent):
    """Gets the rouge-1 precision similarity between a chunk and a sentence from the summary.

    Args:
        scorer: RougeScorer. The scorer to compute the rouge metrics.
        chunk: string. The chunk.
        sent: string. The sentence.

    Returns:
        The rouge-1 precision similarity.
    """
    score = scorer.score(chunk, sent)
    return score["rouge1"].precision


def assign_chunk_target(s_bert_model, dictionary, chunks, summary, word_tok, sent_tok, scorer, max_len, dev,
                        is_extractive):
    """Assigns a target to the chunks.

    Args:
        s_bert_model: SentenceTransformer. The model to get the sentences' embeddings.
        dictionary: Dict. The document dictionary with embeddings and sentences.
        chunks: list. The document's chunks.
        summary: string. The document's summary.
        word_tok: Tokenizer. The sub-words' tokenizer.
        sent_tok: Segmenter. The sentences' tokenizer.
        scorer: RougeScorer. The scorer to compute the rouge metrics.
        max_len: int. The summary's max size.
        dev: device. The device to use for the computation.
        is_extractive: string. The kind of target assignment.

    Returns:
        The chunks' targets.
    """
    if len(chunks) == 1:
        targets = [summary]
    else:
        targets = [""] * len(chunks)  # Initialize empty targets.
        summary_sent = sent_tok.segment(summary)
        if is_extractive:
            for sent in summary_sent:
                scores = [get_rouge1_precision(scorer, chunk, sent) for chunk in chunks]
                # Find and sort the chunks with the highest similarity score.
                indices = [i[0] for i in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
                if max_len is None:
                    targets[indices[0]] += " " + sent
                else:
                    # Assign the sentence to the most similar chunk not already full.
                    for idx in indices:
                        if len(word_tok(targets[idx] + " " + sent)["input_ids"]) < max_len:
                            targets[idx] += " " + sent
                            break
                        else:
                            if len(targets[idx]) == 0:
                                targets[idx] += " " + sent
                                break
        else:
            summ_emb = s_bert_model.encode(summary_sent, convert_to_tensor=True, device=dev)
            summ_dictionary = dict(zip(summary_sent, summ_emb))
            for sent in summary_sent:
                scores = []
                summ_sent_emb = summ_dictionary.get(sent)
                for chunk in chunks:
                    chunk_sent = sent_tok.segment(chunk)
                    doc_sent_embeddings = s_bert_model.encode(chunk_sent, convert_to_tensor=True, device=dev)
                    dictionary = dict(zip(chunk_sent, doc_sent_embeddings))
                    chunk_emb = torch.stack([dictionary.get(s) for s in chunk_sent])
                    # Compute the similarity between the chunk embedding and the current sentence.
                    scores.append(util.pytorch_cos_sim(summ_sent_emb, chunk_emb).mean())
                # Find and sort the chunks with the highest similarity score.
                indices = [i[0] for i in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
                if max_len is None:
                    targets[indices[0]] += " " + sent
                else:
                    # Assign the sentence to the most similar chunk not already full.
                    for idx in indices:
                        if len(word_tok(targets[idx] + " " + sent)["input_ids"]) < max_len:
                            targets[idx] += " " + sent
                            break
                        else:
                            if len(targets[idx]) == 0:
                                targets[idx] += " " + sent
                                break
    return targets


def dataset_segmentation(dataset, s_bert_model, word_tok, sent_tok, scorer, min_len, max_len, max_tar_len, dev,
                         dataset_path, idx_path, no_save, is_extractive, is_paragraph, is_max):
    """Segments each document of the dataset using the text segmentation algorithm and assigns the target to the chunks.

    Args:
        dataset: DataFrame. The dataset to process.
        s_bert_model: SentenceTransformer. The model to get the sentences' embeddings.
        word_tok: Tokenizer. The sub-words' tokenizer.
        sent_tok: Segmenter. The sentences' tokenizer.
        scorer: RougeScorer. The scorer to compute the rouge metrics.
        min_len: int. The chunk's min size.
        max_len: int. The chunk's max size.
        max_tar_len: int. The target's max size.
        dev: device. The device to use for the computation.
        dataset_path: string. The path to save the chunked dataset.
        idx_path: string. The path to save the indices dataset.
        no_save: bool. To save the chunked dataframes.
        is_extractive: string. The kind of target assignment.
        is_paragraph: bool.
        is_max: bool.

    Returns:
        dataframe_chunked: DataFrame. The chunked dataset.
        dataframe_idx: DataFrame. The chunks indices.
    """
    dataset_chunks = []
    dataset_targets = []
    dataset_idx = []
    for i in tqdm(range(len(dataset))):
        doc_text = dataset.iloc[i]["text"]
        doc_summary = dataset.iloc[i]["summary"]
        if is_paragraph:
            doc_chunks = paragraph_segmentation(doc_text, word_tok, sent_tok, min_len, max_len)
        else:
            doc_chunks, dictionary = self_segmentation(doc_text, s_bert_model, word_tok, sent_tok, dev, min_len,
                                                       max_len, is_max)
        dataset_chunks.extend(doc_chunks)
        dataset_idx.append(len(doc_chunks))
        if is_paragraph:
            targets = assign_chunk_target(s_bert_model, None, doc_chunks, doc_summary, word_tok, sent_tok, scorer,
                                          max_tar_len, dev, is_extractive)
        else:
            targets = assign_chunk_target(s_bert_model, dictionary, doc_chunks, doc_summary, word_tok, sent_tok, scorer,
                                          max_tar_len, dev, is_extractive)
        dataset_targets.extend(targets)
    dataframe_chunked = pd.DataFrame(data=list(zip(dataset_chunks, dataset_targets)), columns=["text", "summary"])
    dataframe_idx = pd.DataFrame(dataset_idx, columns=["idx"])
    if not no_save:
        dataframe_chunked.to_csv(dataset_path)
        dataframe_idx.to_csv(idx_path)
        if os.path.isfile(dataset_path) and os.path.isfile(idx_path):
            print("The dataframes have been saved successfully.")
    return dataframe_chunked, dataframe_idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="allenai/led-base-16384", help="The tokenizer checkpoint to use")
    parser.add_argument("--dataset", default="billsum", help="The dataset to use")
    parser.add_argument("--dataset_only", default="", help="To apply the text segmentation for just one dataset")
    parser.add_argument("--is_extractive", default=False, action="store_true", help="To do an extractive assignment")
    parser.add_argument("--is_max", default=False, action="store_true", help="To max the cosine similarities")
    parser.add_argument("--is_paragraph", default=False, action="store_true", help="To do a paragraph segmentation")
    parser.add_argument("--min_input_len", type=int, default=1024, help="The input min size")
    parser.add_argument("--max_input_len", type=int, default=2048, help="The input max size")
    parser.add_argument("--max_output_len", type=int, default=512, help="The output max size")
    parser.add_argument("--model", default="models/metric_learning/legal-bert-base-uncased-triplet", help="The model")
    parser.add_argument("--device", default="cuda", help="The device")
    parser.add_argument("--no_save", default=False, action="store_true", help="To save the chunked dataframes")
    parser.add_argument("--seed", type=int, default=1234, help="The seed")
    parser.add_argument("--toy", type=int, default=0, help="The number of toy examples")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    min_input_length = args.min_input_len
    max_input_length = args.max_input_len
    max_output_length = args.max_output_len
    sentence_bert_model = SentenceTransformer(args.model).to(device)
    sentence_bert_model.max_seq_length = 512
    word_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    sent_tokenizer = pysbd.Segmenter(language="en", clean=True)
    scorer_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    is_extractive = args.is_extractive
    is_paragraph = args.is_paragraph
    is_max = args.is_max

    data_dir = "data/"

    if args.dataset_only != "":
        file_path = data_dir + args.dataset_only
        file_chunked_path = os.path.join(data_dir, args.checkpoint + "_" + args.dataset_only + "_chunked_" +
                                         str(max_input_length) + "_" + str(max_output_length))
        file_chunked_idx_path = os.path.join(data_dir + args.checkpoint + "_" + args.dataset_only + "_chunked_idx_" +
                                             str(max_input_length) + "_" + str(max_output_length))
        dataset_only = pd.read_csv(file_path)
        dataset_chunked, dataset_chunked_idx = \
            dataset_segmentation(dataset_only, sentence_bert_model, word_tokenizer, sent_tokenizer, scorer_rouge,
                                 min_input_length, max_input_length, max_output_length, device,
                                 file_chunked_path, file_chunked_idx_path, args.no_save, is_extractive, is_paragraph)
    else:
        train_file_path = os.path.join(data_dir, args.dataset + "_training_set")
        test_file_path = os.path.join(data_dir, args.dataset + "_test_set")

        if args.model == "nlpaueb/legal-bert-base-uncased":
            loss_used = "legal-bert"
        elif args.model == "bert-base-uncased":
            loss_used = "bert"
        else:
            loss_used = "triplet" if args.model == "models/metric_learning/legal-bert-base-uncased-triplet" or \
                                     "models/metric_learning/scibert-scivocab-uncased-triplet" else \
                        "contrastive"
        name_tok = "led" if args.checkpoint == "allenai/led-base-16384" else "bart"

        train_file_chunked_path = os.path.join(data_dir, name_tok + "_" + args.dataset + "_training_set_chunked_"
                                               + str(min_input_length) + "_" + str(max_input_length) + "_" +
                                               loss_used + "_" + str(is_extractive))
        test_file_chunked_path = os.path.join(data_dir + name_tok + "_" + args.dataset + "_test_set_chunked_" +
                                              str(min_input_length) + "_" + str(max_input_length) + "_" + loss_used +
                                              "_" + str(is_extractive))
        train_file_chunked_idx_path = os.path.join(data_dir + name_tok + "_" + args.dataset +
                                                   "_training_set_chunked_idx_" + str(min_input_length) + "_" +
                                                   str(max_input_length) + "_" + loss_used + "_" + str(is_extractive))
        test_file_chunked_idx_path = os.path.join(data_dir + name_tok + "_" + args.dataset +
                                                  "_test_set_chunked_idx_" + str(min_input_length) + "_" +
                                                  str(max_input_length) + "_" + loss_used + "_" + str(is_extractive))

        if is_paragraph:
            train_file_chunked_path += "_paragraph"
            test_file_chunked_path += "_paragraph"
            train_file_chunked_idx_path += "_paragraph"
            test_file_chunked_idx_path += "_paragraph"

        if is_max:
            train_file_chunked_path += "_max"
            test_file_chunked_path += "_max"
            train_file_chunked_idx_path += "_max"
            test_file_chunked_idx_path += "_max"

        # Download the dataset if not already saved, else just load it.
        if not os.path.isfile(train_file_path) or not os.path.isfile(test_file_path):
            # TODO: adds loading of other datasets (i.e., austlii)
            if args.dataset == "pubmed":
                train_dataset = load_dataset("scientific_papers", args.dataset, split="train").to_pandas()
                test_dataset = load_dataset("scientific_papers", args.dataset, split="test").to_pandas()
                train_dataset = train_dataset.rename(columns={"article": "text", "abstract": "summary"})
                test_dataset = test_dataset.rename(columns={"article": "text", "abstract": "summary"})
            else:
                train_dataset = load_dataset(args.dataset, split="train").to_pandas()
                test_dataset = load_dataset(args.dataset, split="test").to_pandas()
            train_dataset.to_csv(train_file_path)
            test_dataset.to_csv(test_file_path)
        else:
            train_dataset = pd.read_csv(train_file_path)
            test_dataset = pd.read_csv(test_file_path)
            if args.dataset == "pubmed":
                train_dataset = train_dataset.rename(columns={"article": "text", "abstract": "summary"})
                test_dataset = test_dataset.rename(columns={"article": "text", "abstract": "summary"})

        if args.toy > 0:
            train_dataset = train_dataset[:args.toy]
            # test_dataset = test_dataset[:args.toy]

        # Chunks the training set if not already done, else just load it.
        if not os.path.isfile(train_file_chunked_path) or not os.path.isfile(train_file_chunked_idx_path):
            train_dataset_chunked, train_dataset_chunked_idx = \
                dataset_segmentation(train_dataset, sentence_bert_model, word_tokenizer, sent_tokenizer, scorer_rouge,
                                     min_input_length, max_input_length, max_output_length, device,
                                     train_file_chunked_path, train_file_chunked_idx_path, args.no_save, is_extractive,
                                     is_paragraph, is_max)
        else:
            print("Chunked dataset already exists")
            train_dataset_chunked = pd.read_csv(train_file_chunked_path)
            train_dataset_chunked_idx = pd.read_csv(train_file_chunked_idx_path)

        # Chunks the test set if not already done, else just load it.
        if not os.path.isfile(test_file_chunked_path) or not os.path.isfile(test_file_chunked_idx_path):
            test_dataset_chunked, test_dataset_chunked_idx = \
                dataset_segmentation(test_dataset, sentence_bert_model, word_tokenizer, sent_tokenizer, scorer_rouge,
                                     min_input_length, max_input_length, max_output_length, device,
                                     test_file_chunked_path, test_file_chunked_idx_path, args.no_save, is_extractive,
                                     is_paragraph, is_max)
        else:
            test_dataset_chunked = pd.read_csv(test_file_chunked_path)
            test_dataset_chunked_idx = pd.read_csv(test_file_chunked_idx_path)
