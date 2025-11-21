import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_IDX = TOKENIZER.pad_token_id

PROMPT_PREFIX = "Translate the question to SQL: "
MAX_SRC_LEN = 512
MAX_TGT_LEN = 512


class T5SQLDataset(Dataset):
    def __init__(self, nl_list, sql_list):
        self.nl_list = nl_list
        self.sql_list = sql_list

    def __len__(self):
        return len(self.nl_list)

    def __getitem__(self, idx):
        nl = self.nl_list[idx]
        sql = self.sql_list[idx] if self.sql_list is not None else None

        input_text = PROMPT_PREFIX + nl

        enc = TOKENIZER(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SRC_LEN,
        )

        if sql is None:
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor([]), 
            }

        dec = TOKENIZER(
            sql,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_TGT_LEN,
        )

        labels = dec["input_ids"].squeeze(0)
        labels[labels == PAD_IDX] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attn_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_IDX)
    attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)

    if labels[0].numel() == 0:
        labels = torch.empty((len(batch), 1), dtype=torch.long)  # dummy
        labels[:] = -100
    else:
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_ids, attn_mask, labels


def load_lines(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines()]


def load_t5_data(batch_size, test_batch_size, data_folder="data"):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))

    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))

    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    train_loader = DataLoader(
        T5SQLDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        T5SQLDataset(dev_x, dev_y),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        T5SQLDataset(test_x, None),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, dev_loader, test_loader
