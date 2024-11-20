import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.nn.functional as F

# Create output directory if it doesn't exist
output_dir = 'softmaxOutputRatio'
os.makedirs(output_dir, exist_ok=True)

# Set the GPU device (I'm only using one gpu for now)
gpu_id = 0
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if gpu_id >= torch.cuda.device_count():
    raise RuntimeError(f"GPU {gpu_id} not found. Available GPUs: {torch.cuda.device_count()}")

torch.cuda.set_device(gpu_id)
device = f'cuda:{gpu_id}'
print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")

# users can decide where the model comes from
prompt = "Enter '1' or '2' to choose where your model is located: \n    1: use the model from '~/.cache/huggingface/hub' (make sure you have huggingface set up locally)\n    2: use the model from your local directory (make sure you have changed the path in the code to match your files)\n"
option = 0
while (True):
    option = int(input(prompt))
    if option == 1 or option == 2:
        break

if option == 1:
    # Load the tokenizer and model
    model_dir = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
else:
    # Base directory where the model is downloaded 
    # TODO: make sure to change the path here to match your local settings
    BASE_MODEL_PATH = "/home/brian/llamaModels/models--meta-llama--Meta-Llama-3-8B-Instruct"

    # The actual model files should be in a 'snapshots' subdirectory
    snapshots_dir = os.path.join(BASE_MODEL_PATH, "snapshots")
    if os.path.exists(snapshots_dir):
        hash_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if hash_dirs:
            MODEL_PATH = os.path.join(snapshots_dir, hash_dirs[0])
        else:
            MODEL_PATH = BASE_MODEL_PATH
    else:
        MODEL_PATH = BASE_MODEL_PATH

    print(f"Loading model from: {MODEL_PATH}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )

    # Load model across multiple GPUs
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)

# Counter to track softmax operations
softmax_counter = 0
all_ratios = []
non_zero_counts = 0

def save_ratio_to_file(ratio_str, ratio, prefix, size):
    """Save the ratio of non-zero values to a txt file"""
    try:
        filename = f"outputSoftmaxNonZeroRatio.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'a') as f:
            f.write(f"Softmax {prefix}: {ratio_str} ({ratio}) | tensor size: {size}\n")
    except Exception as e:
        print(f"Error saving softmax ratio for {prefix}: {e}")

# Store the original softmax function
_original_softmax = F.softmax

# Create a custom function to replace torch.nn.functional.softmax
def custom_softmax(*args, **kwargs):
    global softmax_counter, all_ratios, non_zero_counts, g_total_values

    # Apply the original softmax function
    output = _original_softmax(*args, **kwargs)

    # Calculate the ratio of non-zero values in the output tensor
    total_values = output.numel()
    non_zero_values = (output != 0).sum().item()
    ratio_str = f"{non_zero_values}/{total_values}"
    ratio = non_zero_values / total_values
    all_ratios.append(ratio)
    non_zero_counts += non_zero_values
    g_total_values = total_values

    # Save the ratio along with the stage to the output file
    save_ratio_to_file(ratio_str, f"{ratio:.6f}", f"op{softmax_counter}", output.size())

    softmax_counter += 1
    return output

# Replace the softmax function in torch.nn.functional
F.softmax = custom_softmax

# Prepare the input data
# Read input from a text file
print("reading in the inputs ...")
with open('input.txt', 'r') as file:
    inputs = [file.read()]

# print("here is the input: ")
# for i in inputs:
#     print("--------------------------")
#     print(i)
#     print("--------------------------")

# Get the context length of the prompt
prompt_context_length = len(tokenizer.encode(inputs[0]))

# Tokenize the inputs without batching
encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True)
input_ids = encoded_inputs['input_ids'].to(device)
attention_mask = encoded_inputs['attention_mask'].to(device)

print("Starting inference...")

# Redirect output to a file
responseFilename = f"response.txt"
responseFilepath = os.path.join(output_dir, responseFilename)
with open(responseFilepath, "w") as response_file:
    try:
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask)
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_context_length = len(tokenizer.encode(decoded_outputs[0]))
        for response in decoded_outputs:
            response_file.write(response + "\n" + "-" * 50 + "\n")

    finally:
        F.softmax = _original_softmax
        print(f"Prompt context length: {prompt_context_length}")
        print(f"Output context length: {output_context_length}")
        print(f"\nSoftmax non-zero value ratio files have been saved to the '{output_dir}' directory.")
        print(f"Tracked {softmax_counter} softmax operations.")

        average_ratio = sum(all_ratios) / len(all_ratios)
        print(f"Average non-zero value ratio: {(non_zero_counts/len(all_ratios)):.4f}/{g_total_values}({100*(average_ratio):.4f}%)")
