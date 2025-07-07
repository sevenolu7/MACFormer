import collections
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class IPEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(IPEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, ip_tensor):
        return self.embedding(ip_tensor)

class IPProcessor(nn.Module):
    def __init__(self, ip_dim=4, embed_dim=128, agg_method='mean'):
        super(IPProcessor, self).__init__()
        self.ip_embedding = IPEmbedding(ip_dim, embed_dim)
        self.agg_method = agg_method

    def forward(self, ip_list):
        ip_embeds = [self.ip_embedding(ip_tensor) for ip_tensor in ip_list]
        if self.agg_method == 'mean':
            return torch.mean(torch.stack(ip_embeds), dim=0)
        elif self.agg_method == 'max':
            return torch.max(torch.stack(ip_embeds), dim=0).values
        else:
            return torch.cat(ip_embeds, dim=-1)


def dataPreprocess_bert_from_csv(filename, input_ids, input_types, input_masks, ip_embeds, label, ip_dim=4):
    """
        Preprocess data from a CSV file containing URLs and labels.

        :param filename: The path to the CSV file.
        :param input_ids: List to store input token IDs.
        :param input_types: List to store segment IDs.
        :param input_masks: List to store attention masks.
        :param label: List to store labels.
        :return: None
        """
    pad_size = 200
    embed_dim = 128
    bert_path = "tacl-bert-base-uncased/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    # IP Processor
    ip_processor = IPProcessor(ip_dim=ip_dim, embed_dim=embed_dim, agg_method='mean')

    df = pd.read_csv(filename)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        url = row['url']
        lbl = row['label'].strip().lower()
        ip_list = row['ip_address'].strip('[]').replace("'", "").split(' ')
        ip_list = [ip.strip() for ip in ip_list if ip]

        # BERT processing for URL
        tokens = tokenizer.tokenize(url)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * len(ids)
        masks = [1] * len(ids)

        # Padding / truncation
        if len(ids) < pad_size:
            pad_len = pad_size - len(ids)
            ids += [0] * pad_len
            types += [1] * pad_len
            masks += [0] * pad_len
        else:
            ids = ids[:pad_size]
            types = types[:pad_size]
            masks = masks[:pad_size]

        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)

        # IP embedding processing
        ip_tensors = [torch.tensor([int(part) for part in ip.split('.')], dtype=torch.float32) for ip in ip_list]
        ip_tensor_stack = torch.stack(ip_tensors)
        ip_embed = ip_processor(ip_tensor_stack)
        ip_embeds.append(ip_embed.tolist())

        # Label processing
        if lbl == "mal":
            label.append([1])
        elif lbl == "benign":
            label.append([0])
        else:
            raise ValueError(f"Unknown label: {lbl}")


def dataPreprocess_bert_from_csv_multi(filename, input_ids, input_types, input_masks, ip_embeds, label, ip_dim=4):
    """
        Preprocess data from a CSV file containing URLs and labels.

        :param filename: The path to the CSV file.
        :param input_ids: List to store input token IDs.
        :param input_types: List to store segment IDs.
        :param input_masks: List to store attention masks.
        :param label: List to store labels.
        :return: None
        """
    pad_size = 200
    embed_dim = 128
    bert_path = "tacl-bert-base-uncased/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    # IP Processor
    ip_processor = IPProcessor(ip_dim=ip_dim, embed_dim=embed_dim, agg_method='mean')

    df = pd.read_csv(filename)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        url = row['url']
        lbl = row['label'].strip().lower()
        ip_list = row['ip_address'].strip('[]').replace("'", "").split(' ')
        ip_list = [ip.strip() for ip in ip_list if ip]

        # BERT processing for URL
        tokens = tokenizer.tokenize(url)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * len(ids)
        masks = [1] * len(ids)

        # Padding / truncation
        if len(ids) < pad_size:
            pad_len = pad_size - len(ids)
            ids += [0] * pad_len
            types += [1] * pad_len
            masks += [0] * pad_len
        else:
            ids = ids[:pad_size]
            types = types[:pad_size]
            masks = masks[:pad_size]

        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)

        # IP embedding processing
        ip_tensors = [torch.tensor([int(part) for part in ip.split('.')], dtype=torch.float32) for ip in ip_list]
        ip_tensor_stack = torch.stack(ip_tensors)
        ip_embed = ip_processor(ip_tensor_stack)
        ip_embeds.append(ip_embed.tolist())

        # Label processing
        if lbl == "mal":
            label.append([1])
        elif lbl == "benign":
            label.append([0])
        elif lbl == "phishing":
            label.append([2])  # phishing 类别为 2
        else:
            raise ValueError(f"Unknown label: {lbl}")


def spiltDataset_bert(input_ids, input_types, input_masks, ip_embeds, label):
    """
        Split the dataset into training and testing sets.

        :param input_ids: List of input character IDs.
        :param input_types: List of segment IDs.
        :param input_masks: List of attention masks.
        :param label: List of labels.
        :return: Split datasets for training and testing.
        """
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)
    np.random.shuffle(random_order)

    split_idx = int(len(input_ids) * 0.8)

    input_ids_train = np.array([input_ids[i] for i in random_order[:split_idx]])
    input_types_train = np.array([input_types[i] for i in random_order[:split_idx]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:split_idx]])
    ip_embeds_train = np.array([ip_embeds[i] for i in random_order[:split_idx]])
    y_train = np.array([label[i] for i in random_order[:split_idx]])

    input_ids_test = np.array([input_ids[i] for i in random_order[split_idx:]])
    input_types_test = np.array([input_types[i] for i in random_order[split_idx:]])
    input_masks_test = np.array([input_masks[i] for i in random_order[split_idx:]])
    ip_embeds_test = np.array([ip_embeds[i] for i in random_order[split_idx:]])
    y_test = np.array([label[i] for i in random_order[split_idx:]])

    return input_ids_train, input_types_train, input_masks_train, ip_embeds_train, y_train, input_ids_test, input_types_test, input_masks_test, ip_embeds_test, y_test
