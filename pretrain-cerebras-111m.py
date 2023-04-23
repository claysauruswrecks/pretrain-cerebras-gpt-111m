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


# Step 1: Load the model and the tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
    model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-111M")
    return tokenizer, model


# Step 2: Load the dataset and tokenize its texts
def load_and_tokenize_dataset(tokenizer):
    # dataset = load_dataset("bigcode/the-stack-smol-xl", split="train")
    dataset = load_dataset("bigcode/the-stack-smol", split="train")
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example["content"]),
        batched=True,
        remove_columns=["content"],
    )
    return tokenized_dataset


# Updated to handle the validation and test datasets
def load_and_tokenize_test_and_validation_datasets(tokenizer):
    validation_dataset = load_dataset("bigcode/the-stack-smol-xl", split="validation")
    test_dataset = load_dataset("bigcode/the-stack-smol-xl", split="test")

    tokenized_validation_dataset = validation_dataset.map(
        lambda example: tokenizer(example["content"]),
        batched=True,
        remove_columns=["content"],
    )

    tokenized_test_dataset = test_dataset.map(
        lambda example: tokenizer(example["content"]),
        batched=True,
        remove_columns=["content"],
    )

    return tokenized_validation_dataset, tokenized_test_dataset


def create_data_loader(tokenized_dataset, tokenizer, batch_size=8):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    data_loader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    return data_loader


def configure_training(output_dir="./pretrained_cerebras/"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    return training_args


# Updated to include tokenized_validation_dataset in Trainer
def create_trainer(
    model, tokenizer, tokenized_dataset, tokenized_validation_dataset, training_args
):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
    )
    return trainer


def quantize_model(
    model: torch.nn.Module, example_input: torch.Tensor
) -> torch.nn.Module:
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def main():
    tokenizer, model = load_model_and_tokenizer()
    tokenized_dataset = load_and_tokenize_dataset(tokenizer)
    (
        tokenized_validation_dataset,
        tokenized_test_dataset,
    ) = load_and_tokenize_test_and_validation_datasets(tokenizer)
    training_args = configure_training()

    trainer = create_trainer(
        model, tokenizer, tokenized_dataset, tokenized_validation_dataset, training_args
    )
    trainer.train()

    trainer.save_model("./pretrained_cerebras/")

    example_input = tokenized_dataset[0]["input_ids"]
    quantized_model = quantize_model(model, example_input)

    torch.save(quantized_model.state_dict(), "./pretrained_cerebras/quantized-model.pt")

    test_results = trainer.evaluate(tokenized_test_dataset)
    print("Test Results:", test_results)

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
