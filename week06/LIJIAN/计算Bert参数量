import torch
from transformers import BertModel

def calculate_bert_parameters(model_name="bert-base-uncased"):
    try:
        model = BertModel.from_pretrained(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    total, trainable = calculate_bert_parameters()
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
