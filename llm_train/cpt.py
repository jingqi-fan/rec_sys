import os

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_PATH = os.getenv("BASE_LLM_MODEL", "Qwen/Qwen2.5-0.5B")
DEFAULT_DATA_PATH = os.getenv(
    "KUAI_RAND_CPT_JSONL",
    os.path.join("artifacts", "kuairand_cpt.jsonl"),
)
DEFAULT_OUTPUT_DIR = os.getenv(
    "KUAI_RAND_CPT_OUTPUT_DIR",
    os.path.join("artifacts", "qwen2.5-0.5b-kuairand-cpt-lora"),
)

MAX_LENGTH = 256
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LOGGING_STEPS = 10
SAVE_STEPS = 100
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def load_cpt_dataset(data_path: str = DEFAULT_DATA_PATH, max_length: int = MAX_LENGTH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CPT jsonl not found: {data_path}")

    dataset = load_dataset("json", data_files={"train": data_path})
    return dataset, max_length


def build_tokenizer(model_path: str = DEFAULT_MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def build_training_arguments(output_dir: str = DEFAULT_OUTPUT_DIR):
    use_bf16 = False
    use_fp16 = False
    try:
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                use_bf16 = True
            else:
                use_fp16 = True
    except Exception:
        pass

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=use_bf16,
        fp16=use_fp16,
    )


def run_cpt(
    model_path: str = DEFAULT_MODEL_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_length: int = MAX_LENGTH,
):
    dataset, max_length = load_cpt_dataset(data_path=data_path, max_length=max_length)
    tokenizer = build_tokenizer(model_path=model_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer=tokenizer, max_length=max_length)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_arguments(output_dir=output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print("CPT training started...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    run_cpt()
