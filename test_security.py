import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import argparse
from datetime import datetime

def load_fine_tuned_model(base_model_id, adapter_path, hf_token):
    """
    Load the fine-tuned model with its LoRA adapter weights.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        use_auth_token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=hf_token
    )
    
    # Load and apply LoRA adapter weights
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=2048):
    """
    Generate a response using the fine-tuned model.
    """
    # Format prompt
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def clean_model_response(response):
    """
    Clean the model response to extract only the relevant information.
    """
    # Remove the instruction part if it appears in the response
    if "[/INST]" in response:
        response = response.split("[/INST]", 1)[1].strip()
    
    # For normal/abnormal classification tasks
    import re
    
    # Check if this is a classification task looking for NORMAL/ABNORMAL labels
    normal_match = re.search(r'\b(normal)\b', response.lower())
    abnormal_match = re.search(r'\b(abnormal)\b', response.lower())
    
    if normal_match:
        return "NORMAL"
    elif abnormal_match:
        return "ABNORMAL"
    
    # For other tasks, try to extract a meaningful response
    # Remove repeating patterns
    lines = response.split('\n')
    unique_lines = []
    
    for line in lines:
        line = line.strip()
        if line and line not in unique_lines:
            unique_lines.append(line)
    
    # Join unique lines, but limit to a reasonable length
    clean_response = '\n'.join(unique_lines[:5])  # Limit to first 5 unique lines
    
    # If still too long, truncate
    if len(clean_response) > 500:
        clean_response = clean_response[:497] + "..."
    
    return clean_response

def load_test_cases(json_file_path):
    """
    Load test cases from a JSON file.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        return test_cases
    except Exception as e:
        print(f"Error loading test cases from {json_file_path}: {str(e)}")
        return None

def save_results_to_file(results, output_file):
    """
    Save results to a file in JSON format.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_file}")

def process_test_cases(test_cases, model, tokenizer, batch_size=5):
    """
    Process all test cases and return results.
    """
    results = []
    total = len(test_cases)
    
    for i in range(0, total, batch_size):
        batch = test_cases[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size}...")
        
        for j, test_case in enumerate(batch, 1):
            instruction = test_case.get("instruction", "")
            if not instruction:
                print(f"  Skipping test case {i+j} - no instruction found.")
                continue
                
            print(f"  Processing test case {i+j}/{total}...")
            print(f"  Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"  Instruction: {instruction}")
            
            # Generate response
            response = generate_response(instruction, model, tokenizer)
            
            # Clean the response
            cleaned_response = clean_model_response(response)
            
            # Store result
            results.append({
                "instruction": test_case.get("instruction", ""),
                "output": cleaned_response
            })

            print(f"  Response generated successfully for test case {i+j}.")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned model with test cases from a JSON file")
    parser.add_argument("--test_file", required=True, help="Path to the JSON file containing test cases")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model ID")
    parser.add_argument("--adapter_path", default="./llama3_securitymodel/checkpoint-8000", help="Path to LoRA adapter weights")
    parser.add_argument("--hf_token", default="", help="Hugging Face token") # Update with your own HF_TOKEN here
    parser.add_argument("--output_dir", default="generation_results", help="Directory to save results")
    parser.add_argument("--output_file", help="Specific output filename (optional, otherwise timestamp will be used)")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of test cases to process in each batch")
    args = parser.parse_args()
    
    try:
        # Create an output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate output filename
        if args.output_file:
            output_file = os.path.join(args.output_dir, args.output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
        
        # Load test cases
        print(f"Loading test cases from {args.test_file}...")
        test_cases = load_test_cases(args.test_file)
        if not test_cases:
            print("No test cases found. Exiting.")
            return
        
        if not isinstance(test_cases, list):
            print(f"Test file should contain a JSON array. Found {type(test_cases)}. Exiting.")
            return
            
        print(f"Loaded {len(test_cases)} test cases.")
            
        print("Loading model and tokenizer...")
        # Load model
        model, tokenizer = load_fine_tuned_model(args.base_model, args.adapter_path, args.hf_token)
        print("Model and tokenizer loaded successfully!")
        
        # Process test cases and get results
        results = process_test_cases(test_cases, model, tokenizer, batch_size=args.batch_size)
        
        # Save results
        save_results_to_file(results, output_file)
        print(f"\nAll test cases processed successfully! Results saved to {output_file}")
        
    except Exception as e:
        print(f"\nAn error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
