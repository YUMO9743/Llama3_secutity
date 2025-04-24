import os
import transformers
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from torch import cuda
from peft import LoraConfig
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Mobile LLaMA for specific tasks")
    parser.add_argument('--output_dir', type=str, default="../llama3_finetuned",
                        help='Directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--optim', type=str, default="paged_adamw_32bit",
                        help='Optimizer type')
    parser.add_argument('--logging_steps', type=int, default=200,
                        help='Logging interval')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=0.3,
                        help='Max gradient norm')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum number of training steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout rate')
    parser.add_argument('--lora_r', type=int, default=64,
                        help='LoRA r parameter')
    parser.add_argument('--num_train_epochs', type=float, default=3.0,
                        help='Number of training epochs')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use fp16 training')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='Learning rate scheduler type')

    return parser.parse_args()

def main():
    os.environ["WANDB_DISABLED"] = "true"
    args = get_args()
    hf_token = ""  # Update with your own HF_TOKEN
    model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    
    # Load JSON data and convert to Dataset
    with open("../training_data/security_dataset.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)
    
    # Convert the list to a dataset
    dataset = Dataset.from_dict({
        "instruction": [item["instruction"] for item in data_list],
        "output": [item["output"] for item in data_list]
    })

    model = load_model(model_id, hf_token)
    tokenizer = load_tokenizer(model_id, hf_token)
    trainer = setup_training(dataset, tokenizer, model, args)
    trainer.train()

def load_model(model_id, hf_auth):
    model_config = AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)
    bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print(f"Model loaded on {device}")
    return model

def load_tokenizer(model_id, hf_auth):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def setup_training(dataset, tokenizer, model, args):
    # Split the dataset
    train_val = dataset.train_test_split(test_size=1300, shuffle=True, seed=42)

    def generate_and_tokenize_prompt(data_points):
        full_prompts = []
        for instruction, output in zip(data_points["instruction"], data_points["output"]):
            prompt = f"""### Instruction: {instruction}
            ### Response: {output}"""
            full_prompts.append(prompt)
        
        # Add text field for SFTTrainer
        data_points["text"] = full_prompts
        return data_points
        # Tokenize all prompts at once
        tokenized = tokenizer(
            full_prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors=None
        )
        
        # Make sure all sequences are of equal length by padding to max_length
        max_length = max(len(ids) for ids in tokenized["input_ids"])
        
        # Pad all sequences to the same length
        for i in range(len(tokenized["input_ids"])):
            if len(tokenized["input_ids"][i]) < max_length:
                padding_length = max_length - len(tokenized["input_ids"][i])
                tokenized["input_ids"][i].extend([tokenizer.pad_token_id] * padding_length)
                tokenized["attention_mask"][i].extend([0] * padding_length)
        
        # Add labels
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        return tokenized
    
    # Process the datasets
    train_data = train_val["train"].map(
        generate_and_tokenize_prompt,
        batched=True,
        batch_size=100,
        remove_columns=train_val["train"].column_names
    )
    
    val_data = train_val["test"].map(
        generate_and_tokenize_prompt,
        batched=True,
        batch_size=100,
        remove_columns=train_val["test"].column_names
    )

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        group_by_length=True,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    # PEFT config
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",  # Specify the text field
        max_seq_length=1024
        # data_collator=transformers.DataCollatorForLanguageModeling(
        #     tokenizer=tokenizer,
        #     mlm=False
        # )
    )

    return trainer

if __name__ == '__main__':
    main()
