import os
import argparse
from tqdm import tqdm

import torch
from transformers import T5TokenizerFast

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_IDX = TOKENIZER.pad_token_id 

BOS_ID = TOKENIZER.convert_tokens_to_ids("<extra_id_0>")


def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop")

    # decoding 超参数
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.0)

    # model / optimizer
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--learning_rate", type=float, default=1e-3)  # 1e-1
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_epochs", type=int, default=0)
    parser.add_argument("--max_n_epochs", type=int, default=20)
    parser.add_argument("--patience_epochs", type=int, default=5)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="experiment")

    # dataloader
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    return parser.parse_args()


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    n_steps = 0

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    for input_ids, attn_mask, labels in tqdm(train_loader, desc="Train"):
        optimizer.zero_grad()

        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,   #huggingfacedebug
        )
        logits = outputs.logits 
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        mask = labels != -100

        if mask.any():
            loss = criterion(logits[mask], labels[mask])
        else:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)



def eval_epoch(args, model, dev_loader,
               gt_sql_path, model_sql_path,
               gt_record_path, model_record_path):
    model.eval()

    total_loss = 0.0
    n_steps = 0
    generated_queries = []

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    with torch.no_grad():
        for input_ids, attn_mask, labels in tqdm(dev_loader, desc="Dev"):
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            logits = outputs.logits

            vocab_size = logits.size(-1)
            logits = logits.reshape(-1, vocab_size)
            labels = labels.reshape(-1)
            mask = labels != -100

            if mask.any():
                loss = criterion(logits[mask], labels[mask])
                total_loss += loss.item()
                n_steps += 1

            
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_start_token_id=BOS_ID, 
                max_length=300, 
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
            )
            decoded = [TOKENIZER.decode(g, skip_special_tokens=True) for g in gen_ids]
            generated_queries.extend(decoded)

    avg_loss = total_loss / max(1, n_steps)

    # 保存 SQL+records
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    #print(f"Saved dev SQL to {model_sql_path}")
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    error_count = sum(1 for m in error_msgs if m != "")
    error_rate = error_count / len(error_msgs) if error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    generated_queries = []

    with torch.no_grad():
        for input_ids, attn_mask, _ in tqdm(test_loader, desc="Test"):
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_start_token_id=BOS_ID,   # ★ 与dev 保持一致
                max_length=300,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
            )
            decoded = [TOKENIZER.decode(g, skip_special_tokens=True) for g in gen_ids]
            generated_queries.extend(decoded)

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    print(f"Saved test SQL to {model_sql_path}")
    print(f"Saved test records to {model_record_path}")


# ---------------- MAIN LOOP ----------------
def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size
    )

    # checkpoint 目录
    model_type = "ft" if args.finetune else "scr"
    save_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # 如果有 last_model.pt 就接着训练，否则从头
    last_ckpt = os.path.join(save_dir, "last_model.pt")
    if os.path.exists(last_ckpt):
        print(f"Resuming from {last_ckpt}")
        model = initialize_model(args)
        state = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"])
    else:
        print("Starting from scratch.")
        print("Loading pretrained model google-t5/t5-small...")
        model = initialize_model(args)

    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    best_f1 = -1.0
    no_improve = 0

    dev_sql_path = os.path.join("data", "dev.sql")
    dev_record_gt = os.path.join("records", "ground_truth_dev.pkl")
    dev_sql_out = os.path.join("results", f"t5_{model_type}_ft_experiment_dev.sql")
    dev_record_out = os.path.join("records", f"t5_{model_type}_ft_experiment_dev.pkl")

    # ---- train ----
    for epoch in range(args.max_n_epochs):
        train_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"[Epoch {epoch}] Train loss = {train_loss:.4f}")

        eval_loss, f1, em_r, em_sql, err_r = eval_epoch(
            args, model, dev_loader,
            dev_sql_path, dev_sql_out,
            dev_record_gt, dev_record_out
        )
        print(f"[Dev] loss={eval_loss:.4f} F1={f1:.4f} EM={em_r:.4f} SQL-EM={em_sql:.4f}")

        # 存 last
        save_model(save_dir, model, best=False)

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            save_model(save_dir, model, best=True)
            print(f"New best F1 = {best_f1:.4f}")
        else:
            no_improve += 1

        if no_improve >= args.patience_epochs:
            print("Early stopping.")
            break

    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    eval_loss, f1, em_r, em_sql, err_r = eval_epoch(
        args, model, dev_loader,
        dev_sql_path, dev_sql_out,
        dev_record_gt, dev_record_out
    )
    print(f"[Dev-final] loss={eval_loss:.4f} F1={f1:.4f} EM={em_r:.4f} SQL-EM={em_sql:.4f}")

    # test
    #test_sql_out = os.path.join("results", "t5_ft_experiment_test.sql")
    #test_record_out = os.path.join("records", "t5_ft_experiment_test.pkl")
    test_sql_out = os.path.join("results", f"t5_{model_type}_experiment_test.sql")
    test_record_out = os.path.join("records", f"t5_{model_type}_experiment_test.pkl")
    test_inference(args, model, test_loader, test_sql_out, test_record_out)


if __name__ == "__main__":
    main()
