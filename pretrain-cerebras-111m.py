import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.integrations import TensorBoardCallback


# Step 1: Load the model and the tokenizer
def load_model_and_tokenizer():
    # Load the tokenizer and the model for GPT-111M
    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
    model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-111M")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# Step 2: Load the dataset and tokenize its texts
def load_dataset_without_tokenization():
    # Load the dataset with the specified split
    dataset = load_dataset("bigcode/the-stack-smol", split="train")
    return dataset


def create_validation_and_test_splits(
    raw_dataset, tokenizer, val_ratio=0.1, test_ratio=0.1, max_sequence_length=2048
):
    val_size = int(len(raw_dataset) * val_ratio)
    test_size = int(len(raw_dataset) * test_ratio)

    # Shuffle the raw dataset
    raw_dataset = raw_dataset.shuffle(seed=42)

    # Get validation, test, and the remaining train dataset
    val_dataset = raw_dataset.select(range(val_size))
    test_dataset = raw_dataset.select(range(val_size, val_size + test_size))
    train_dataset = raw_dataset.select(range(val_size + test_size, len(raw_dataset)))

    # Tokenize the train, validation, and test dataset
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["content"], truncation=True, padding=False, max_length=max_sequence_length
        ),
        batched=True,
        remove_columns=["content"],
    )
    tokenized_val_dataset = val_dataset.map(
        lambda x: tokenizer(
            x["content"], truncation=True, padding=False, max_length=max_sequence_length
        ),
        batched=True,
        remove_columns=["content"],
    )
    tokenized_test_dataset = test_dataset.map(
        lambda x: tokenizer(
            x["content"], truncation=True, padding=False, max_length=max_sequence_length
        ),
        batched=True,
        remove_columns=["content"],
    )

    return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset


def create_data_loader(tokenized_dataset, tokenizer, batch_size=4):
    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    # Create a DataLoader with the specified batch size and data collator
    data_loader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    return data_loader


def configure_training(output_dir="./pretrained_cerebras/"):
    # Configure the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=1_000,
        save_total_limit=5,
        prediction_loss_only=False,
        optim="adamw_torch",
        evaluation_strategy="steps",
        logging_dir="./logs",
        log_level="info",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        log_on_each_node=True,
        logging_first_step=True,
        logging_steps=1_000,
    )
    return training_args


# Updated to include tokenized_validation_dataset in Trainer
def create_trainer(
    model, tokenizer, tokenized_dataset, tokenized_validation_dataset, training_args
):
    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    # Create a Trainer with the specified model, training arguments, datasets and tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
    )
    return trainer


def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def main():
    # torch.set_num_threads(24)
    # torch.set_num_interop_threads(24)
    # torch.set_num_threads(6)
    # torch.set_num_interop_threads(6)
    print(
        f"threads: (num_threads, num_interop_threads) ({torch.get_num_threads()}, {torch.get_num_interop_threads()})"
    )
    tokenizer, model = load_model_and_tokenizer()
    # tokenizer, original_model = load_model_and_tokenizer()
    # model = quantize_model(original_model)
    raw_dataset = load_dataset_without_tokenization()
    (
        tokenized_train_dataset,
        tokenized_val_dataset,
        tokenized_test_dataset,
    ) = create_validation_and_test_splits(raw_dataset, tokenizer)

    training_args = configure_training()

    trainer = create_trainer(
        model, tokenizer, tokenized_train_dataset, tokenized_val_dataset, training_args
    )
    trainer.add_callback(TensorBoardCallback())

    trainer.train()

    trainer.save_model("./pretrained_cerebras/")

    # example_input = tokenized_train_dataset[0]["input_ids"]
    # quantized_model = quantize_model(model, example_input)
    # quantized_model = quantize_model(model)

    # torch.save(
    #     quantized_model.state_dict(),
    #     "./pretrained_cerebras/quantized-pretrained-model.pt",
    # )

    val_results = trainer.evaluate(tokenized_val_dataset)
    print("Validation Results:", val_results)

    text = "Generative AI is "
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[
        0
    ]
    print(generated_text["generated_text"])

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=50,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(text_output[0])


if __name__ == "__main__":
    main()
