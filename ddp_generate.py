import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# Define DDP setup function
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "147.47.68.91"
    os.environ["MASTER_PORT"] = "12357"
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def load_model_and_data():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset


    tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-small-augm")
    model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-small-augm")

    # Load and preprocess the dataset
    dataset = load_dataset("language-plus-molecules/LPM-24_train")

    def tokenize_function(examples):
        # Add prompt
        prompt = "Caption the following SMILES: "
        examples['molecule'] = [prompt + molecule for molecule in examples['molecule']]
        
        # Tokenize the input text
        model_inputs = tokenizer(examples["molecule"], padding="max_length", truncation=True)
        
        """ # Tokenize the target text
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["caption"], padding="max_length", truncation=True)
    
        # Add labels to the model inputs
        model_inputs["labels"] = labels["input_ids"] """
        
        return model_inputs

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    ### Extract test dataset
    test_dataset = tokenized_datasets["split_valid"].shuffle(seed=42)

    return model, tokenizer, test_dataset

# Define inference function
def inference(model, tokenizer, dataloader, rank):
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in dataloader:
            #tokenized_prompt = tokenizer("Caption the following SMILES: ", return_tensors="pt", padding="max_length", truncation=True)
            # input_ids of prompt and molecule
            # input_ids = torch.cat((tokenized_prompt["input_ids"].unsqueeze(0).expand(len(batch["input_ids"]), -1), batch["input_ids"]), dim=1).to(rank)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Generate SMILES captions
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=5)
            predicted_caption = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            # Append pairs to the predictions list
            predictions.extend(zip(batch["molecule"], predicted_caption))
    return predictions

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, output_file_path: str, batch_size: int):
    ddp_setup(rank, world_size)

    # Load model
    model, tokenizer, test_dataset = load_model_and_data()

    # Load test dataloader
    test_dataloader = prepare_dataloader(test_dataset, batch_size)

    # Perform inference
    predictions = inference(model, tokenizer, test_dataloader, rank)

    # Save predictions to file
    with open(output_file_path, "w") as f:
        for pair in predictions:
            f.write("\t".join(pair) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMILES Captioning Inference with DDP")
    parser.add_argument("output_file_path", type=str, help="Path to save the predictions")
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args.output_file_path, args.batch_size), nprocs=world_size)
