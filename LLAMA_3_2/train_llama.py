import cv2
import os
import subprocess
import torch
import json
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from torch.utils.data import Dataset
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig
import requests

def get_best_gpu():
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", 
            shell=True
        ).decode("utf-8").strip()
        gpus = [line.split(", ") for line in result.split("\n")]
        best_gpu = max(gpus, key=lambda x: int(x[1]))[0]  # Select GPU with the most free memory
        return int(best_gpu)
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return None

# Automatically choose the best GPU
best_gpu = get_best_gpu()

if best_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Using the best GPU: {best_gpu}")
else:
    print("No GPU available; defaulting to CPU.")

# Login to Hugging Face Hub
login(
    token="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",  # Replace with your token
    add_to_git_credential=True
)

# Define a custom dataset class
class VisionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define a function to validate and load images using OpenCV
def load_image_with_opencv(image_path):
    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_path}")
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping invalid image file: {image_path}")
            return None
        # Convert BGR to RGB as required by most image models
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Skipping invalid image file: {image_path}, Error: {e}")
        return None
    return image


def format_data_with_check(sample):
    image_path = sample.get("image", None)

    if not os.path.exists(image_path):
        print(f"Skipping sample due to missing image path: {sample}")
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample['conversations'][0]['value'][8:]},
                    {"type": "image", "image": image_path}
                ],
            },
            {"role": "assistant", "content": sample['conversations'][1]['value']},
        ]
    }

# Load dataset from JSON
file_url = 'https://raw.githubusercontent.com/Gokhman07/MEDICAL_LLAVA_DATASET/refs/heads/master/data5.json'
response = requests.get(file_url)
response.raise_for_status()
data = response.json()
formatted_data = [format_data_with_check(sample) for sample in data]
formatted_data = [d for d in formatted_data if d is not None]

# Wrap the dataset in the custom PyTorch Dataset class
dataset = VisionDataset(formatted_data)

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model and processor setup
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map={"": device},  # Map the model to the specified device
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)

# Training configuration
args = SFTConfig(
    output_dir="scratch/medical_dataset_file/fine-tuned-visionllama",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=torch.cuda.is_available(),  # Use bf16 only if supported
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="tensorboard",
    dataset_kwargs={"skip_prepare_dataset": True},
    max_seq_length=1024  # Explicitly set max sequence length
)
args.remove_unused_columns = False

# Define a custom collator function
def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens in the labels
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

# PEFT Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
    peft_config=peft_config
)

# Start training
trainer.train()
trainer.model.save_pretrained('scratch/medical_dataset_file/fine-tuned-visionllama-lora')

