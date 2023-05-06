import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


# Step 1: Load the model and the tokenizer
def load_model_and_tokenizer():
    # Load the tokenizer and the model for GPT-111M
    tokenizer = AutoTokenizer.from_pretrained(
        "claysauruswrecks/cerebras-gpt-111m-pretrain-stack-smol-0-15k-chkp"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "claysauruswrecks/cerebras-gpt-111m-pretrain-stack-smol-0-15k-chkp"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def main():
    torch.set_num_threads(24)
    torch.set_num_interop_threads(24)
    # Uncomment for (default) A100 GPU usage
    # torch.set_num_threads(6)
    # torch.set_num_interop_threads(6)
    print(
        f"threads: (num_threads, num_interop_threads) ({torch.get_num_threads()}, {torch.get_num_interop_threads()})"
    )
    tokenizer, model = load_model_and_tokenizer()

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
