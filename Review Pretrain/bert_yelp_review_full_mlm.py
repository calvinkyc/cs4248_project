from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer

USE_CHUNK = True
MAX_LENGTH = 128
print(f"USE_CHUNK: {USE_CHUNK} {MAX_LENGTH if USE_CHUNK else ''}")

yelp = load_dataset("yelp_review_full", keep_in_memory=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_func(examples):
    result = tokenizer(examples["text"], truncation=not USE_CHUNK, max_length=MAX_LENGTH if not USE_CHUNK else None)
    return result


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // MAX_LENGTH) * MAX_LENGTH
    # Split by chunks of max_len
    result = {
        k: [t[i: i + MAX_LENGTH] for i in range(0, total_length, MAX_LENGTH)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


yelp_tokenized = yelp.map(tokenize_func, batched=True, remove_columns=["text", "label"])
if USE_CHUNK:
    yelp_tokenized = yelp_tokenized.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")

training_args = TrainingArguments(
    output_dir="./pretrain_results/yelp_review_full/",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=256,
    fp16=True,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    save_total_limit=1,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=yelp_tokenized["train"],
    eval_dataset=yelp_tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print(trainer.evaluate())
trainer.train()
print(trainer.evaluate())
