import torch
from tqdm import tqdm

from functions.bert_metrics import get_accuracy_from_logits

def evaluate(model, criterion, dataloader, device):
	model.eval()
	mean_acc, mean_loss, count = 0, 0, 0
	with torch.no_grad():
		for input_ids, attention_mask, image, labels in tqdm(dataloader, desc="Evaluating"):
			input_ids, attention_mask, image, labels = input_ids.to(device), attention_mask.to(device), image.to(device, dtype=torch.float), labels.to(device)
			logits = model(input_ids, attention_mask, image)
			mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
			mean_acc += get_accuracy_from_logits(logits, labels)
			count += 1
	return mean_acc / count, mean_loss / count