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
    # Load the AutoTokenizer and AutoModelForCausalLM to work with the Cerebras-GPT-111M model
    tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
    model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-111M")
    return tokenizer, model


# Step 2: Load the dataset and tokenize its texts
def load_and_tokenize_dataset(tokenizer):
    # Load the wikitext dataset and tokenize its text using the provided tokenizer
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example["text"]),
        batched=True,
        remove_columns=["text"],
    )
    return tokenized_dataset


# Step 3: Create a DataLoader to handle batches during training
def create_data_loader(tokenized_dataset, tokenizer, batch_size=8):
    # Initialize a DataCollatorForLanguageModeling with the given tokenizer and no masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    # Create a DataLoader with the tokenized dataset and the data collator to generate batches
    data_loader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    return data_loader


# Step 4: Configure the training process
def configure_training(output_dir="./pretrained_cerebras/"):
    # Set up the training arguments with output directory, epochs, batch size, etc.
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


# Step 5: Create a Trainer instance with the provided configurations
def create_trainer(model, tokenizer, tokenized_dataset, training_args):
    # Initialize a DataCollatorForLanguageModeling with the given tokenizer for training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    # Create a Trainer with the configurations, model, data collator, dataset, and tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    return trainer


# Step 6: Quantize the model to reduce its size and improve inference performance
def quantize_model(
    model: torch.nn.Module, example_input: torch.Tensor
) -> torch.nn.Module:
    # Perform dynamic quantization on the model, specifying the layers to quantize
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


# Step 7: Main function to run the full process
def main():
    # Load the model and tokenizer, prepare the dataset and training configurations
    tokenizer, model = load_model_and_tokenizer()
    tokenized_dataset = load_and_tokenize_dataset(tokenizer)
    training_args = configure_training()

    # Create a Trainer instance and perform the training
    trainer = create_trainer(model, tokenizer, tokenized_dataset, training_args)
    trainer.train()

    # Save the original trained model
    trainer.save_model("./pretrained_cerebras/")

    # Quantize the trained model for better inference performance
    example_input = tokenized_dataset[0][
        "input_ids"
    ]  # Get an example input from the dataset
    quantized_model = quantize_model(model, example_input)

    # Save the quantized model
    torch.save(quantized_model.state_dict(), "./pretrained_cerebras/quantized-model.pt")

    # Example usage of the model with Hugging Face Pipelines
    text = "Generative AI is "
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[
        0
    ]
    print(generated_text["generated_text"])

    # Example usage of the model with its `generate()` method
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
