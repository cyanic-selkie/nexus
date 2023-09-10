import argparse
from model import instantiate_model
from transformers import TrainingArguments, Trainer
from dataset import DataCollatorForEL, get_dataset
from datasets import disable_caching

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--continue-training", action="store_true")

    parser.add_argument("--embedding-size", type=int, required=True)
    parser.add_argument('--language', action='append')

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps",
                        type=int,
                        required=True)

    args = parser.parse_args()

    # disable_caching()

    model, tokenizer = instantiate_model(args.checkpoint, args.embedding_size)
    dataset, train_total, validation_total = get_dataset(
        tokenizer, args.language)

    data_collator = DataCollatorForEL(tokenizer, args.embedding_size)

    args = TrainingArguments(
        output_dir=f"models/{args.name}",
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        lr_scheduler_type="constant_with_warmup",
        weight_decay=0.01,
        report_to="wandb",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        logging_steps=20,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if args.continue_training:
        trainer.train(args.checkpoint)
    else:
        trainer.train()
