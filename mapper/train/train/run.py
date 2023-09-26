import argparse
import os
from clearml import Task
from embeddings import Embeddings
from model import instantiate_model
from transformers import TrainingArguments, Trainer
from dataset import DataCollatorForEL, get_dataset_wikianc, get_dataset_cronel, get_dataset_conll, get_dataset_cronel_conll

if __name__ == "__main__":
    os.environ["CLEARML_TASK_NO_REUSE"] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument('--language', action='append')
    parser.add_argument('--dataset',
                        choices=[
                            "wikianc", "cronel", "aida-conll-yago-wikidata",
                            "cronel+aida-conll-yago-wikidata"
                        ],
                        default="wikianc")
    parser.add_argument('--embeddings-dir', type=str, required=True)

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps",
                        type=int,
                        required=True)

    args = parser.parse_args()

    embeddings = Embeddings(args.embeddings_dir)

    model, tokenizer = instantiate_model(args.checkpoint, args.embedding_size)

    if args.dataset == "cronel":
        dataset, train_total, validation_total = get_dataset_cronel(
            tokenizer, embeddings)
    elif args.dataset == "aida-conll-yago-wikidata":
        dataset, train_total, validation_total = get_dataset_conll(
            tokenizer, embeddings)
    elif args.dataset == "cronel+aida-conll-yago-wikidata":
        dataset, train_total, validation_total = get_dataset_conll(
            tokenizer, embeddings)
    else:
        dataset, train_total, validation_total = get_dataset_wikianc(
            tokenizer, embeddings, args.language)

    data_collator = DataCollatorForEL(tokenizer, args.embedding_size)

    training_args = TrainingArguments(
        output_dir=f"models/{args.name}",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # num_train_epochs=args.epochs,
        lr_scheduler_type="constant_with_warmup",
        weight_decay=0.01,
        report_to="clearml",
        save_strategy="epoch",
        # save_steps=1000,
        # save_total_limit=5,
        logging_steps=100,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=(train_total // args.batch_size) * args.epochs,
    )

    task = Task.init(project_name="nexus", task_name=args.name)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
