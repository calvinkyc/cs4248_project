from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer

MAX_LENGTH = 512

ds_raw = load_dataset("amazon_polarity", keep_in_memory=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_func(examples):
    result = tokenizer(examples["content"], truncation=True, max_length=MAX_LENGTH)
    return result


ds = ds_raw.map(tokenize_func, batched=True, remove_columns=["title", "content"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./pretrain_results/amazon_polarity_cls/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    fp16=True,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    save_total_limit=1,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print(trainer.evaluate())
trainer.train()
print(trainer.evaluate())
